import numpy as np

import torch
import torch.nn.functional as F

from instance.core.config import cfg
from instance.modeling.mask_head.inference import masks_results
from instance.modeling.keypoint_head.inference import get_final_preds
from instance.modeling.parsing_head.inference import parsing_results
from instance.modeling.uv_head.inference import uvs_results
from instance.utils.transforms import flip_back, xywh_to_xyxy
from utils.data.structures.densepose_uv import flip_uv_featuremap


def conv_body_inference(model, inputs):
    features = [im_conv_body_net(model, inputs)]
    if cfg.TEST.AUG.H_FLIP:
        features.append(im_conv_body_net(model, inputs, flip=True))

    for scale in cfg.TEST.AUG.SCALES:
        features.append(im_conv_body_net(model, inputs, scale=scale))
        if cfg.TEST.AUG.H_FLIP:
            features.append(im_conv_body_net(model, inputs, scale=scale, flip=True))

    return features


def mask_inference(model, features):
    _idx = 0
    size = [cfg.TEST.SCALE[1], cfg.TEST.SCALE[0]]
    mask_results = []
    results = model.mask_net(features[_idx]).cpu().numpy()
    _idx += 1
    mask_results.append(results)
    if cfg.TEST.AUG.H_FLIP:
        results_hf = model.mask_net(features[_idx]).cpu().numpy()
        _idx += 1
        results_hf = flip_back(results_hf)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if cfg.TEST.AUG.SHIFT_HEATMAP:
            results_hf[:, :, :, 1:] = results_hf[:, :, :, 0:-1]
        mask_results.append(results_hf)

    for scale in cfg.TEST.AUG.SCALES:
        results_scl = model.mask_net(features[_idx])
        _idx += 1
        results_scl = F.interpolate(results_scl, size=size, mode="bilinear", align_corners=False)
        mask_results.append(results_scl.cpu().numpy())
        if cfg.TEST.AUG.H_FLIP:
            results_scl_hf = model.mask_net(features[_idx])
            _idx += 1
            results_scl_hf = F.interpolate(results_scl_hf, size=size, mode="bilinear", align_corners=False)
            results_scl_hf = flip_back(results_scl_hf.cpu().numpy())
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.AUG.SHIFT_HEATMAP:
                results_scl_hf[:, :, :, 1:] = results_scl_hf[:, :, :, 0:-1]
            mask_results.append(results_scl_hf)

    mask_results = np.mean(mask_results, axis=0)
    return mask_results


def keypoint_inference(model, features):
    flip_map = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    _idx = 0
    size = [cfg.KEYPOINT.HEATMAP_SIZE[1], cfg.KEYPOINT.HEATMAP_SIZE[0]]
    keypoint_results = []
    results = model.keypoint_net(features[_idx]).cpu().numpy()
    _idx += 1
    keypoint_results.append(results)
    if cfg.TEST.AUG.H_FLIP:
        results_hf = model.keypoint_net(features[_idx]).cpu().numpy()
        _idx += 1
        results_hf = flip_back(results_hf, flip_map)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if cfg.TEST.AUG.SHIFT_HEATMAP:
            results_hf[:, :, :, 1:] = results_hf[:, :, :, 0:-1]
        keypoint_results.append(results_hf)

    for scale in cfg.TEST.AUG.SCALES:
        results_scl = model.keypoint_net(features[_idx])
        _idx += 1
        results_scl = F.interpolate(results_scl, size=size, mode="bilinear", align_corners=False)
        keypoint_results.append(results_scl.cpu().numpy())
        if cfg.TEST.AUG.H_FLIP:
            results_scl_hf = model.keypoint_net(features[_idx])
            _idx += 1
            results_scl_hf = F.interpolate(results_scl_hf, size=size, mode="bilinear", align_corners=False)
            results_scl_hf = flip_back(results_scl_hf.cpu().numpy(), flip_map)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.AUG.SHIFT_HEATMAP:
                results_scl_hf[:, :, :, 1:] = results_scl_hf[:, :, :, 0:-1]
            keypoint_results.append(results_scl_hf)

    keypoint_results = np.mean(keypoint_results, axis=0)
    return keypoint_results


def parsing_inference(model, features):
    if 'CIHP' in cfg.TEST.DATASETS[0]:
        flip_map = ([14, 15], [16, 17], [18, 19])
    elif 'MHP-v2' in cfg.TEST.DATASETS[0]:
        flip_map = ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])
    elif 'ATR' in cfg.TEST.DATASETS[0]:
        flip_map = ([9, 10], [12, 13], [14, 15])
    elif 'LIP' in cfg.TEST.DATASETS[0]:
        flip_map = ([14, 15], [16, 17], [18, 19])
    else:
        flip_map = ()

    _idx = 0
    size = [cfg.TEST.SCALE[1], cfg.TEST.SCALE[0]]
    parsing_results = []
    parsing_scores = []
    outputs = model.parsing_net(features[_idx])
    results = outputs['parsings'].cpu().numpy()
    scores = outputs['parsing_scores']
    if scores is not None:
        scores = scores.cpu().numpy()
    _idx += 1
    parsing_results.append(results)
    parsing_scores.append(scores)
    if cfg.TEST.AUG.H_FLIP:
        results_hf = model.parsing_net(features[_idx])['parsings'].cpu().numpy()
        scores_hf = model.parsing_net(features[_idx])['parsing_scores']
        if scores_hf is not None:
            scores_hf = scores_hf.cpu().numpy()
        _idx += 1
        results_hf = flip_back(results_hf, flip_map)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if cfg.TEST.AUG.SHIFT_HEATMAP:
            results_hf[:, :, :, 1:] = results_hf[:, :, :, 0:-1]
        parsing_results.append(results_hf)
        parsing_scores.append(scores_hf)

    for scale in cfg.TEST.AUG.SCALES:
        results_scl = model.parsing_net(features[_idx])['parsings']
        _idx += 1
        results_scl = F.interpolate(results_scl, size=size, mode="bilinear", align_corners=False)
        parsing_results.append(results_scl.cpu().numpy())
        if cfg.TEST.AUG.H_FLIP:
            results_scl_hf = model.parsing_net(features[_idx])['parsings']
            _idx += 1
            results_scl_hf = F.interpolate(results_scl_hf, size=size, mode="bilinear", align_corners=False)
            results_scl_hf = flip_back(results_scl_hf.cpu().numpy(), flip_map)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.AUG.SHIFT_HEATMAP:
                results_scl_hf[:, :, :, 1:] = results_scl_hf[:, :, :, 0:-1]
            parsing_results.append(results_scl_hf)

    parsing_results = np.mean(parsing_results, axis=0)
    parsing_scores = np.mean(parsing_scores, axis=0) if scores is not None else None
    return parsing_results, parsing_scores


def uv_inference(model, features):
    _idx = 0
    size = [cfg.TEST.SCALE[1], cfg.TEST.SCALE[0]]
    uv_results = [[], [], [], []]
    results = model.uv_net(features[_idx])
    _idx += 1
    results = [result.cpu().numpy() for result in results]
    add_uv_results(uv_results, results)
    if cfg.TEST.AUG.H_FLIP:
        results_hf = model.uv_net(features[_idx])
        _idx += 1
        results_hf = [result.cpu().numpy() for result in results_hf]
        results_hf = flip_uv_featuremap(results_hf)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if cfg.TEST.AUG.SHIFT_HEATMAP:
            results_hf[:, :, :, 1:] = results_hf[:, :, :, 0:-1]
        add_uv_results(uv_results, results_hf)

    for scale in cfg.TEST.AUG.SCALES:
        results_scl = model.uv_net(features[_idx])
        _idx += 1
        results_scl = [F.interpolate(result, size=size, mode="bilinear", align_corners=False)
                       for result in results_scl]
        results_scl = [result.cpu().numpy() for result in results_scl]
        add_uv_results(uv_results, results_scl)
        if cfg.TEST.AUG.H_FLIP:
            results_scl_hf = model.uv_net(features[_idx])
            _idx += 1
            results_scl_hf = [
                F.interpolate(result, size=size, mode="bilinear", align_corners=False) for result in results_scl_hf
            ]
            results_scl_hf = [result.cpu().numpy() for result in results_scl_hf]
            results_scl_hf = flip_uv_featuremap(results_scl_hf)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.AUG.SHIFT_HEATMAP:
                results_scl_hf[:, :, :, 1:] = results_scl_hf[:, :, :, 0:-1]
            add_uv_results(uv_results, results_scl_hf)

    _uv_results = []
    for i in range(4):
        _uv_results.append(np.mean(uv_results[i], axis=0))
    return _uv_results


def im_conv_body_net(model, inputs, scale=None, flip=False):
    if scale is not None:
        size = [scale[1], scale[0]]
        inputs = F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)
    if flip:
        inputs = inputs.flip(3)
    result = model.conv_body_net(inputs)
    return result


def add_uv_results(all_results, results):
    for i in range(4):
        all_results[i].append(results[i])


def post_processing(result, targets, val_set):
    c = targets['center'].numpy()
    s = targets['scale'].numpy()
    image_id = targets['image_id'].numpy()
    boxes = targets['bbox'].numpy()
    score = targets['score'].numpy()
    im_per_batch = image_id.shape[0]
    ins_info = np.zeros((im_per_batch, 10))
    image_info = [(entry['height'], entry['width']) for entry in val_set.coco.loadImgs(image_id)]

    # double check this all_boxes parts
    ins_info[:, 0] = image_id
    ins_info[:, 1] = score
    ins_info[:, 2] = np.prod(s * 200, 1)

    if 'mask_class' in targets.keys():
        classes = targets['mask_class'].numpy().astype(np.int)
        ins_info[:, 3] = classes
    else:
        classes = np.ones(im_per_batch).astype(np.int)
        ins_info[:, 3] = classes

    boxes_xyxy = xywh_to_xyxy(boxes)
    boxes_score = np.hstack((boxes_xyxy, score[:, np.newaxis])).astype(np.float32, copy=False)

    if cfg.MODEL.MASK_ON:
        masks = masks_results(result['mask'], image_info, c, s, classes)
    else:
        masks = []

    if cfg.MODEL.KEYPOINT_ON:
        keyps = np.zeros((im_per_batch, cfg.KEYPOINT.NUM_JOINTS, 3), dtype=np.float32)
        preds, maxvals = get_final_preds(result['keypoints'], c, s)
        keyps[:, :, 0:2] = preds[:, :, 0:2]
        keyps[:, :, 2:3] = maxvals
        keyps = list(keyps)
    else:
        keyps = []

    if cfg.MODEL.PARSING_ON:
        parss, pars_scores = parsing_results(result['parsing'], image_info, c, s)
        if result['parsing_score'] is None:
            ins_info[:, 9] = pars_scores
        else:
            ins_info[:, 9] = result['parsing_score']
    else:
        parss = []

    if cfg.MODEL.UV_ON:
        uvs, uv_scores = uvs_results(result['uv'], image_info, boxes, c, s)
        ins_info[:, 4:8] = boxes
        ins_info[:, 8] = uv_scores
    else:
        uvs = []

    ins_info = list(ins_info)

    return ins_info, boxes_score, classes, masks, keyps, parss, uvs
