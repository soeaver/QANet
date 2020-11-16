import json
import os
import pickle
import shutil
import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.sparse import csr_matrix

from lib.data.evaluation.densepose_eval import DensePoseEvaluator
from lib.data.evaluation.parsing_eval import ParsingEvaluator, generate_parsing_result
from lib.utils.visualizer import Visualizer

from qanet.datasets import dataset_catalog


class Evaluation(object):
    def __init__(self, cfg, training=False):
        """
        Evaluation
        :param cfg: config
        """
        self.cfg = cfg
        self.training = training
        self.iou_types = ()
        self.pet_results = {}
        self.all_iou_types = ("segm", "parsing", "keypoints", "uv")

    def parsing_eval(self, iou_type, dataset, output_folder):
        """Interface of Parsing evaluation
        """
        gt_im_dir = dataset_catalog.get_im_dir(self.cfg.TEST.DATASETS[0])
        metrics = self.cfg.PARSING.METRICS if not self.training else ['mIoU', ]
        pet_eval = ParsingEvaluator(
            dataset, self.pet_results[iou_type], gt_im_dir, output_folder, self.cfg.PARSING.SCORE_THRESH,
            self.cfg.PARSING.NUM_PARSING, metrics=metrics
        )
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mIoU = pet_eval.stats['mIoU']
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mIoU

    def coco_eval(self, iou_type, dataset, output_folder):
        """Interface of COCO evaluation
        """
        file_path = os.path.join(output_folder, iou_type + ".json")
        pet_eval = evaluate_on_coco(self.cfg, dataset.coco, self.pet_results[iou_type], file_path, iou_type)
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mAP = 0.0 if 'lvis' in self.cfg.TEST.DATASETS[0] else pet_eval.stats[0]
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mAP

    def post_processing(self, results, targets, image_ids, dataset):
        """Prepare results by preparing function of each task
        """
        num_im = len(image_ids)
        eval_results = []
        ims_results = []
        prepare_funcs = []
        prepare_funcs = self.get_prepare_func(prepare_funcs)
        for prepare_func in prepare_funcs:
            prepared_result = self.prepare_results(results, targets, image_ids, dataset, prepare_func)
            if prepared_result is not None:
                assert len(prepared_result) >= 2
                eval_results.append(prepared_result[0])
                # box results include box and label
                ims_results.extend(prepared_result[1:])
            else:
                eval_results.append([])
                ims_results.append([None for _ in range(num_im)])
        if self.cfg.VIS.ENABLED:
            self.vis_processing(ims_results, targets, image_ids, dataset)
        return eval_results

    def vis_processing(self, ims_results, targets, image_ids, dataset):
        ims_dets = [
            np.hstack((target.im_bbox.numpy(), target.scores.numpy()[:, np.newaxis])).astype(np.float32, copy=False)
            for target in targets
        ]
        ims_labels = [target.labels.tolist() for target in targets]

        ims_masks, ims_kpts, ims_parss, ims_uvs = ims_results
        for k, idx in enumerate(image_ids):
            if len(ims_dets[k]) == 0:
                continue

            im = dataset.pull_image(idx)
            visualizer = Visualizer(self.cfg.VIS, im, dataset=dataset)
            im_name = dataset.get_img_info(image_ids[k])['file_name']
            vis_im = visualizer.vis_preds(
                boxes=ims_dets[k],
                classes=ims_labels[k],
                masks=ims_masks[k],
                keypoints=ims_kpts[k],
                parsings=ims_parss[k],
                uvs=ims_uvs[k],
            )
            cv2.imwrite(os.path.join(self.cfg.CKPT, 'vis', '{}'.format(im_name)), vis_im)

    def evaluation(self, dataset, all_results):
        """Eval results by iou types
        """
        output_folder = os.path.join(self.cfg.CKPT, 'test')
        self.get_pet_results(all_results)

        for iou_type in self.iou_types:
            if iou_type == "parsing":
                eval_result = self.parsing_eval(iou_type, dataset, output_folder)
            elif iou_type in self.all_iou_types:
                eval_result = self.coco_eval(iou_type, dataset, output_folder)
            else:
                raise KeyError("{} is not supported!".format(iou_type))
        if self.cfg.CLEAN_UP:  # clean up all the test files
            shutil.rmtree(output_folder)
        return eval_result

    def prepare_results(self, results, targets, image_ids, dataset, prepare_func=None):
        """Prepare result of each task for evaluation
        """
        if prepare_func is not None:
            return prepare_func(self.cfg, results, targets, image_ids, dataset)
        else:
            return None

    def get_pet_results(self, all_results):
        """Get preparing function of each task
        """
        all_masks, all_keyps, all_parss, all_uvs = all_results
        if self.cfg.MODEL.MASK_ON:
            self.iou_types = self.iou_types + ("segm",)
            self.pet_results["segm"] = all_masks
        if self.cfg.MODEL.KEYPOINT_ON:
            self.iou_types = self.iou_types + ("keypoints",)
            self.pet_results['keypoints'] = all_keyps
        if self.cfg.MODEL.PARSING_ON:
            self.iou_types = self.iou_types + ("parsing",)
            self.pet_results['parsing'] = all_parss
        if self.cfg.MODEL.UV_ON:
            self.iou_types = self.iou_types + ("uv",)
            self.pet_results['uv'] = all_uvs

    def get_prepare_func(self, prepare_funcs):
        """Get preparing function of each task
        """
        if self.cfg.MODEL.MASK_ON:
            prepare_funcs.append(prepare_mask_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.KEYPOINT_ON:
            prepare_funcs.append(prepare_keypoint_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.PARSING_ON:
            prepare_funcs.append(prepare_parsing_results)
        else:
            prepare_funcs.append(None)

        if self.cfg.MODEL.UV_ON:
            prepare_funcs.append(prepare_uv_results)
        else:
            prepare_funcs.append(None)

        return prepare_funcs


def prepare_mask_results(cfg, results, targets, image_ids, dataset):
    mask_results = []
    ims_masks = []

    if 'mask' not in results.keys():
        return mask_results, ims_masks

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_masks.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        bitmasks = results['mask']['ims_bitmasks'][i][:, :image_height, :image_width]
        rles = []
        for j in range(len(bitmasks)):
            # Too slow.
            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(bitmasks[j][:, :, np.newaxis], dtype=np.uint8, order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            rles.append(rle)
        # calculating quality scores
        mask_bbox_scores = target.scores
        mask_iou_scores = results['mask']['mask_iou_scores'][i]
        mask_pixel_scores = results['mask']['mask_pixle_scores'][i]
        alpha, beta, gamma = cfg.MASK.QUALITY_WEIGHTS
        _dot = torch.pow(mask_bbox_scores, alpha) * torch.pow(mask_iou_scores, beta) * \
               torch.pow(mask_pixel_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_masks.append(rles)
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        mask_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return mask_results, ims_masks


def prepare_keypoint_results(cfg, results, targets, image_ids, dataset):
    kpt_results = []
    ims_kpts = []

    if 'keypoints' not in results.keys():
        return kpt_results, ims_kpts

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_kpts.append(None)
            continue

        keypoints = results['keypoints']['ims_kpts'][i].numpy()
        # calculating quality scores
        kpt_bbox_scores = target.scores
        kpt_iou_scores = results['keypoints']['kpt_iou_scores'][i]
        kpt_pixle_scores = results['keypoints']['kpt_pixle_scores'][i]
        alpha, beta, gamma = cfg.KEYPOINT.QUALITY_WEIGHTS
        _dot = torch.pow(kpt_bbox_scores, alpha) * torch.pow(kpt_iou_scores, beta) * \
               torch.pow(kpt_pixle_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_kpts.append(keypoints.transpose((0, 2, 1)))
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        kpt_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "keypoints": keypoint.flatten().tolist(),
                    "score": scores[k]
                }
                for k, keypoint in enumerate(keypoints)
            ]
        )
    return kpt_results, ims_kpts


def prepare_parsing_results(cfg, results, targets, image_ids, dataset):
    pars_results = []
    ims_parss = []
    output_folder = os.path.join(cfg.CKPT, 'test')

    if 'parsing' not in results.keys():
        return pars_results, ims_parss

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_parss.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        parsings = results['parsing']['ims_parsings'][i][:, :image_height, :image_width]
        # calculating quality scores
        parsing_bbox_scores = target.scores
        parsing_iou_scores = results['parsing']['parsing_iou_scores'][i]
        parsing_instance_pixel_scores = results['parsing']['parsing_instance_pixel_scores'][i]
        parsing_part_pixel_scores = results['parsing']['parsing_part_pixel_scores'][i]
        alpha, beta, gamma = cfg.PARSING.QUALITY_WEIGHTS
        instance_dot = torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta) * \
                       torch.pow(parsing_instance_pixel_scores, gamma)
        instance_scores = torch.pow(instance_dot, 1. / sum((alpha, beta, gamma))).tolist()
        part_dot = torch.stack([torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta)] *
                               (cfg.PARSING.NUM_PARSING - 1), dim=1) * torch.pow(parsing_part_pixel_scores, gamma)
        part_scores = torch.pow(part_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_parss.append(parsings.numpy())
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        parsings, instance_scores = generate_parsing_result(
            parsings, instance_scores, part_scores, parsing_bbox_scores.tolist(), semseg=None, img_info=img_info,
            output_folder=output_folder, score_thresh=cfg.PARSING.SCORE_THRESH,
            semseg_thresh=cfg.PARSING.SEMSEG_SCORE_THRESH, parsing_nms_thres=cfg.PARSING.PARSING_NMS_TH,
            num_parsing=cfg.PARSING.NUM_PARSING
        )
        pars_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "parsing": csr_matrix(parsing),
                    "score": instance_scores[k]
                }
                for k, parsing in enumerate(parsings)
            ]
        )
    return pars_results, ims_parss


def prepare_uv_results(cfg, results, targets, image_ids, dataset):
    uvs_results = []
    ims_uvs = []

    if 'uv' not in results.keys():
        return uvs_results, ims_uvs

    ims_Index_UV = results['uv']['ims_Index_UV']
    ims_U_uv = results['uv']['ims_U_uv']
    ims_V_uv = results['uv']['ims_V_uv']
    h, w = ims_Index_UV[0].shape[1:]

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_uvs.append(None)
            continue
        uvs = []
        Index_UV = ims_Index_UV[i].numpy()
        U_uv = ims_U_uv[i].numpy()
        V_uv = ims_V_uv[i].numpy()

        for ind, entry in enumerate(target.im_bbox.numpy()):
            x1 = int(entry[0])
            y1 = int(entry[1])
            x2 = int(entry[2])
            y2 = int(entry[3])

            output = np.zeros([3, int(y2 - y1), int(x2 - x1)], dtype=np.float32)
            output[0] = Index_UV[ind][y1:y2, x1:x2]

            outputU = np.zeros([h, w], dtype=np.float32)
            outputV = np.zeros([h, w], dtype=np.float32)

            for part_id in range(1, cfg.UV.NUM_PATCHES + 1):
                CurrentU = U_uv[ind][part_id]
                CurrentV = V_uv[ind][part_id]
                outputU[Index_UV[ind] == part_id] = CurrentU[Index_UV[ind] == part_id]
                outputV[Index_UV[ind] == part_id] = CurrentV[Index_UV[ind] == part_id]
            output[1] = outputU[y1:y2, x1:x2]
            output[2] = outputV[y1:y2, x1:x2]
            uvs.append(output.copy())

        # calculating quality scores
        uv_bbox_scores = target.scores
        uv_iou_scores = results['uv']['uv_iou_scores'][i]
        uv_pixel_scores = results['uv']['uv_pixel_scores'][i]
        alpha, beta, gamma = cfg.UV.QUALITY_WEIGHTS
        _dot = torch.pow(uv_bbox_scores, alpha) * torch.pow(uv_iou_scores, beta) * \
               torch.pow(uv_pixel_scores, gamma)
        scores = torch.pow(_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_uvs.append(uvs)
        for uv in uvs:
            uv[1:3, :, :] = uv[1:3, :, :] * 255
        box_dets = target.im_bbox.int()
        xs = box_dets[:, 0].tolist()
        ys = box_dets[:, 1].tolist()
        ws = (box_dets[:, 2] - box_dets[:, 0]).tolist()
        hs = (box_dets[:, 3] - box_dets[:, 1]).tolist()
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        uvs_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "uv": uv.astype(np.uint8),
                    "bbox": [xs[k], ys[k], ws[k], hs[k]],
                    "score": scores[k]
                }
                for k, uv in enumerate(uvs)
            ]
        )

    return uvs_results, ims_uvs


def evaluate_on_coco(cfg, coco_gt, coco_results, json_result_file, iou_type):
    if iou_type != "uv":
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)
        if cfg.MODEL.HIER_ON and iou_type == "bbox":
            box_results = get_box_result(cfg)
            coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
            coco_gt = coco_gt.loadRes(box_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type, True)
        else:
            if 'lvis' in cfg.TEST.DATASETS[0]:
                from lvis import LVIS, LVISResults, LVISEval
                lvis_gt = LVIS(dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0]))
                lvis_results = LVISResults(lvis_gt, coco_results)
                coco_eval = LVISEval(lvis_gt, lvis_results, iou_type)
            else:
                coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
                coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    else:
        pkl_result_file = json_result_file.replace('.json', '.pkl')
        with open(pkl_result_file, 'wb') as f:
            pickle.dump(coco_results, f, 2)
        if cfg.TEST.DATASETS[0].find('test') > -1:
            return
        eval_data_dir = cfg.DATA_DIR + '/coco/annotations/DensePoseData/eval_data/'
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = DensePoseEvaluator(coco_gt, coco_dt, iou_type, eval_data_dir, calc_mode=cfg.UV.CALC_MODE)
    return coco_eval
