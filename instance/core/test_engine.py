import os
import cv2
import yaml
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils.measure import measure_model
from utils.misc import mkdir_p, logging_rank, save_object
from utils.checkpointer import get_weights, load_weights
from utils.net import convert_bn2affine_model
from utils.logger import TestingLogger
from utils.timer import Timer
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils

from instance.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
from instance.modeling.model_builder import Generalized_CNN
from instance.datasets import build_dataset, make_test_data_loader, evaluation
from instance.core.test import conv_body_inference, mask_inference, keypoint_inference
from instance.core.test import parsing_inference, uv_inference, qanet_inference, post_processing


def run_inference(args, ind_range=None, multi_gpu_testing=False):
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            return test_net_on_dataset(args, multi_gpu=multi_gpu_testing)
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            return test_net(args, ind_range=ind_range)

    all_results = result_getter()
    return all_results


def test_net_on_dataset(args, multi_gpu=False):
    """Run inference on a dataset."""
    val_set = build_dataset(cfg.TEST.DATASETS, is_train=False)
    
    # Calculate FLOPs & Param
    n_flops, n_convops, n_params = -1e6, -1e6, -1e6
    if cfg.CALC_FLOPS:
        model = Generalized_CNN()
        model.eval()
        logging_rank(model)
        n_flops, n_convops, n_params = measure_model(model, cfg.TEST.SCALE[0], cfg.TEST.SCALE[1])
        del model
    logging_rank('FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
                 format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6))

    total_timer = Timer()
    total_timer.tic()
    if multi_gpu:
        num_images = len(val_set)
        all_info, all_masks, all_keyps, all_parss, all_uvs = multi_gpu_test_net_on_dataset(args, num_images)
    else:
        all_info, all_masks, all_keyps, all_parss, all_uvs = test_net(args)
    total_timer.toc(average=False)
    logging_rank('Total inference time: {:.3f}s'.format(total_timer.average_time), local_rank=0)

    return evaluation(val_set, all_info, all_masks, all_keyps, all_parss, all_uvs, cfg.CLEAN_UP)


def multi_gpu_test_net_on_dataset(args, num_images):
    """Multi-gpu inference on a dataset."""
    binary_dir = os.getcwd()
    binary = os.path.join(binary_dir, args.test_net_file + '.py')
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel('instances', num_images, binary, cfg, cfg.CKPT)

    # Collate the results from each subprocess
    all_info = []
    all_masks = []
    all_keyps = []
    all_parss = []
    all_uvs = []

    for ins_data in outputs:
        all_info += ins_data['all_info']
        all_masks += ins_data['all_masks']
        all_keyps += ins_data['all_keyps']
        all_parss += ins_data['all_parss']
        all_uvs += ins_data['all_uvs']

    ins_file = os.path.join(cfg.CKPT, 'test', 'instances.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_info=all_info,
            all_masks=all_masks,
            all_keyps=all_keyps,
            all_parss=all_parss,
            all_uvs=all_uvs,
            cfg=cfg_yaml
        ), ins_file
    )
    logging_rank('Wrote instances to: {}'.format(os.path.abspath(ins_file)), local_rank=0)

    return all_info, all_masks, all_keyps, all_parss, all_uvs


def test_net(args, ind_range=None):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    val_set = build_dataset(cfg.TEST.DATASETS, is_train=False)
    logger = TestingLogger(args.cfg_file.split('/')[-1], log_period=int(np.ceil(10 / cfg.TEST.IMS_PER_GPU)))

    if ind_range is not None:
        start_ind, end_ind = ind_range
    else:
        start_ind = 0
        end_ind = len(val_set)
    model = initialize_model_from_cfg()
    all_info, all_masks, all_keyps, all_parss, all_uvs = test(model, val_set, start_ind, end_ind, logger)
    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        ins_name = 'instances_range_%s_%s.pkl' % tuple(ind_range)
    else:
        ins_name = 'instances.pkl'
    ins_file = os.path.join(cfg.CKPT, 'test', ins_name)
    save_object(
        dict(
            all_info=all_info,
            all_masks=all_masks,
            all_keyps=all_keyps,
            all_parss=all_parss,
            all_uvs=all_uvs,
            cfg=cfg_yaml
        ), ins_file
    )
    logging_rank('Wrote instances to: {}'.format(os.path.abspath(ins_file)), local_rank=0)
    return all_info, all_masks, all_keyps, all_parss, all_uvs


def test(model, val_set, start_ind, end_ind, logger):
    # Switch to evaluate mode
    device = torch.device(cfg.DEVICE)
    loader = make_test_data_loader(val_set, start_ind, end_ind)

    total_num_images = len(val_set)
    all_info = []
    all_masks = []
    all_keyps = []
    all_parss = []
    all_uvs = []
    visualize_imgs = []

    with torch.no_grad():
        logger.iter_tic()
        logger.data_tic()
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            logger.data_toc()

            logger.infer_tic()
            result = {}
            features = conv_body_inference(model, inputs)
            if cfg.MODEL.MASK_ON:
                result['mask'] = mask_inference(model, features)
            if cfg.MODEL.KEYPOINT_ON:
                result['keypoints'] = keypoint_inference(model, features)
            if cfg.MODEL.PARSING_ON:
                result['parsing'], result['parsing_score'] = parsing_inference(model, features)
            if cfg.MODEL.UV_ON:
                result['uv'] = uv_inference(model, features)
            if cfg.MODEL.QANET_ON:
                result['parsing'], result['parsing_score'] = qanet_inference(model, features)
            logger.infer_toc()

            logger.post_tic()
            ins_info, boxes, classes, masks, keyps, parss, uvs = post_processing(result, targets, val_set)
            if cfg.VIS.ENABLED:
                for k in range(len(ins_info)):
                    filename = val_set.coco.loadImgs([ins_info[k][0]])[0]['file_name']
                    img_path = os.path.join(val_set.root, filename)
                    if img_path not in visualize_imgs:
                        im = cv2.imread(img_path)
                        visualize_imgs.append(img_path)
                    else:
                        im = cv2.imread(os.path.join(cfg.CKPT, 'vis/') + filename)

                    ims_masks = [masks[k]] if cfg.MODEL.MASK_ON else None
                    ims_keyps = [keyps[k].transpose()] if cfg.MODEL.KEYPOINT_ON else None
                    ims_parss = [parss[k]] if cfg.MODEL.PARSING_ON or cfg.MODEL.QANET_ON else None
                    ims_uvs = [uvs[k]] if cfg.MODEL.UV_ON else None
                    vis_im = vis_utils.vis_one_image_opencv(
                        im,
                        cfg.VIS,
                        boxes[k:k+1],
                        [classes[k]],
                        masks=ims_masks,
                        keypoints=ims_keyps,
                        parsing=ims_parss,
                        uv=ims_uvs,
                        dataset=val_set
                    )
                    cv2.imwrite(os.path.join(cfg.CKPT, 'vis', '{}'.format(filename)), vis_im)
            all_info += ins_info
            all_masks += masks
            all_keyps += keyps
            all_parss += parss
            all_uvs += uvs
            logger.post_toc()
            logger.iter_toc()
            logger.log_stats(i * cfg.TEST.IMS_PER_GPU + start_ind, start_ind, end_ind, total_num_images)

            logger.iter_tic()
            logger.data_tic()
    return all_info, all_masks, all_keyps, all_parss, all_uvs


def initialize_model_from_cfg():
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    # Create model
    model = Generalized_CNN()
    # Load trained model
    cfg.TEST.WEIGHTS = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, cfg.TEST.WEIGHTS)
    if cfg.MODEL.BATCH_NORM == 'freeze':
        model = convert_bn2affine_model(model)
    model.eval()
    model.to(torch.device(cfg.DEVICE))
    if cfg.DEVICE == 'cuda':
        cudnn.benchmark = True

    return model
