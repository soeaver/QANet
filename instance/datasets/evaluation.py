import os
import numpy as np
import json
import pickle
import shutil
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pet.utils.misc import logging_rank
from pet.utils.data.evaluation.densepose_cocoeval import denseposeCOCOeval
from pet.utils.data.evaluation.parsing_eval import parsing_png, evaluate_parsing
from pet.instance.core.config import cfg
from pet.instance.datasets import dataset_catalog
from pet.instance.utils.misc import oks_nms


def evaluation(dataset, all_info, all_masks, all_keyps, all_parss, all_uvs, clean_up=True):
    output_folder = os.path.join(cfg.CKPT, 'test')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    joint_name = ''.join(list(cfg.TEST.DATASETS))

    results = {}
    if cfg.MODEL.KEYPOINT_ON:
        results['keypoints'] = prepare_for_coco_keypoint(dataset, all_info, all_keyps)
        res_file = os.path.join(output_folder, 'keypoints_%s_results.json' % joint_name)
        res = evaluate_predictions_on_coco(dataset.coco, results['keypoints'], res_file, 'keypoints')
        logging_rank('Evaluating keypoints is done!')
    if cfg.MODEL.PARSING_ON:
        results['parsing'] = prepare_for_parsing(dataset, all_info, all_parss)
        eval_ap = cfg.PARSING.EVAL_AP
        num_parsing = cfg.PARSING.NUM_PARSING
        assert len(cfg.TEST.DATASETS) == 1, 'Parsing only support one dataset now'
        im_dir = dataset_catalog.get_im_dir(cfg.TEST.DATASETS[0])
        ann_fn = dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0])
        res = evaluate_parsing(
            results['parsing'], eval_ap, cfg.PARSING.SCORE_THRESH, num_parsing, im_dir, ann_fn, output_folder
        )
        logging_rank('Evaluating parsing is done!')
    if cfg.MODEL.MASK_ON:
        results['mask'] = prepare_for_coco_mask(dataset, all_info, all_masks)
        res_file = os.path.join(output_folder, 'mask_%s_results.json' % joint_name)
        res = evaluate_predictions_on_coco(dataset.coco, results['mask'], res_file, 'segm')
        logging_rank('Evaluating mask is done!')
    if cfg.MODEL.UV_ON:
        results['UV'] = prepare_for_coco_densepose(dataset, all_info, all_uvs)
        res_file = os.path.join(output_folder, 'uv_%s_results.pkl' % joint_name)
        res = evaluate_predictions_on_coco(dataset.coco, results['UV'], res_file, 'uv')
        logging_rank('Evaluating uv is done!')
    if clean_up:    # clean up all the results files
        shutil.rmtree(output_folder)
        

def prepare_for_coco_mask(dataset, all_info, all_masks):
    coco_results = []
    for idx, mask in enumerate(all_masks):
        category_id = dataset.contiguous_category_id_to_json_id[all_info[idx][3]]
        coco_results.append({
            'image_id': all_info[idx][0],
            'category_id': category_id,
            'segmentation': mask,
            'score': all_info[idx][1]
        })

    return coco_results


def prepare_for_coco_keypoint(dataset, all_info, all_keyps):
    # person x (keypoints)
    _kpts = []
    for idx, kpt in enumerate(all_keyps):
        _kpts.append({
            'keypoints': kpt,
            'image': all_info[idx][0],
            'score': all_info[idx][1],
            'area': all_info[idx][2]
        })
    # image x person x (keypoints)
    kpts = defaultdict(list)
    for kpt in _kpts:
        kpts[kpt['image']].append(kpt)

    # re-scoring and oks nms
    num_joints = cfg.KEYPOINT.NUM_JOINTS
    oks_thre = cfg.KEYPOINT.OKS_THRESH
    in_vis_thre = cfg.KEYPOINT.IN_VIS_THRESH
    oks_nmsed_kpts = []
    for img in kpts.keys():
        img_kpts = kpts[img]
        for n_p in img_kpts:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, num_joints):
                t_s = n_p['keypoints'][n_jt][2]
                if t_s > in_vis_thre:
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # re-scoring
            n_p['score'] = kpt_score * box_score
        keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    category_ids = dataset.coco.getCatIds()
    categories = [c['name'] for c in dataset.coco.loadCats(category_ids)]
    _class_to_coco_cat_id = dict(zip(categories, dataset.coco.getCatIds()))
    person_id = _class_to_coco_cat_id['person']

    coco_results = []
    for img_kpts in oks_nmsed_kpts:
        for kpt in img_kpts:
            keypoint = kpt['keypoints'].flatten().tolist()
            coco_results.append({
                'image_id': kpt['image'],
                'category_id': person_id,
                'keypoints': keypoint,
                'score': kpt['score']
            })

    return coco_results


def prepare_for_parsing(dataset, all_info, all_parss):
    # person x (parsing)
    output_folder = os.path.join(cfg.CKPT, 'test')
    _parss = []
    for idx, parsing in enumerate(all_parss):
        score = all_info[idx][1] if cfg.PARSING.USE_BBOX_CONF else all_info[idx][1] * all_info[idx][9]

        _parss.append({
            'parsing': parsing,
            'image': all_info[idx][0],
            'score': score,
            'area': all_info[idx][2]
        })
    # image x person x (parsing)
    parss = defaultdict(list)
    for parsing in _parss:
        parss[parsing['image']].append(parsing)

    all_parsing = []
    all_scores = []
    for img in tqdm(parss.keys(), desc='Create parsing image'):
        img_parsing = []
        img_score = []
        img_parss = parss[img]
        for n_p in img_parss:
            img_parsing.append(n_p['parsing'])
            img_score.append(n_p['score'])
        img_info = dataset.coco.loadImgs(int(img_parss[0]['image']))[0]
        img_parsing = np.array(img_parsing)
        img_score = np.array(img_score)
        parsing_png(img_parsing, img_score, cfg.PARSING.SEMSEG_SCORE_THRESH, img_info, output_folder)
        all_parsing.append(img_parsing)
        all_scores.append(img_score)

    results = [all_parsing, all_scores]
    return results


def prepare_for_coco_densepose(dataset, all_info, all_uvs):
    coco_results = []
    for idx, uv in enumerate(all_uvs):
        uv[1:3, :, :] = uv[1:3, :, :] * 255
        score = all_info[idx][1] if cfg.UV.USE_BBOX_CONF else all_info[idx][1] * all_info[idx][8]
        coco_results.append({
            'image_id': all_info[idx][0],
            'category_id': 1,
            'uv': uv.astype(np.uint8),
            'bbox': [int(all_info[idx][4]), int(all_info[idx][5]), int(all_info[idx][6]), int(all_info[idx][7])],
            'score': score
        })

    return coco_results


def evaluate_predictions_on_coco(coco_gt, coco_results, result_file, iou_type="bbox"):
    if iou_type != 'uv':
        with open(result_file, "w") as f:
            json.dump(coco_results, f)
        coco_dt = coco_gt.loadRes(str(result_file)) if coco_results else COCO()
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
    else:
        calc_mode = 'GPSm' if cfg.UV.GPSM_ON else 'GPS'
        with open(result_file, 'wb') as f:
            pickle.dump(coco_results, f, 2)
        evalDataDir = os.path.dirname(__file__) + cfg.DATA_DIR + '/DensePoseData/eval_data/'
        coco_dt = coco_gt.loadRes(coco_results)
        test_sigma = 0.255
        coco_eval = denseposeCOCOeval(evalDataDir, coco_gt, coco_dt, iou_type, test_sigma)
        coco_eval.evaluate(calc_mode=calc_mode)
    coco_eval.accumulate()

    coco_eval.summarize()
    return coco_eval
