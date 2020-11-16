import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data.structures.densepose_uv import flip_uv_prob
from lib.data.structures.image_list import to_image_list
from lib.data.structures.instance_box import instancebox_split
from lib.data.structures.keypoint import flip_keypoints_prob
from lib.data.structures.parsing import flip_parsing_prob
from lib.ops import roi_align_rotated
from lib.utils.comm import is_main_process
from lib.utils.misc import mkdir_p

from qanet.datasets.dataset_catalog import get_extra_fields
from qanet.modeling.keypoint_head.inference import get_keypoints_results
from qanet.modeling.mask_head.inference import get_mask_results
from qanet.modeling.parsing_head.inference import get_parsing_results
from qanet.modeling.uv_head.inference import get_uv_results


class TestEngine(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.result = {}
        self.features = []
        self.extra_fields = get_extra_fields(self.cfg.TEST.DATASETS[0])
        self.preprocess_inputs = PreprocessInputs(self.cfg)

        if is_main_process():
            if not os.path.isdir(os.path.join(self.cfg.CKPT, 'test')):
                mkdir_p(os.path.join(self.cfg.CKPT, 'test'))
            if self.cfg.VIS.ENABLED:
                if not os.path.exists(os.path.join(self.cfg.CKPT, 'vis')):
                    mkdir_p(os.path.join(self.cfg.CKPT, 'vis'))

    def __call__(self, inputs, targets):
        if self.cfg.TEST.INSTANCES_PER_BATCH == -1:
            self.conv_body_inference(inputs, targets)
            if len(self.features) == 0:
                return self.result
            self.mask_inference(targets) if self.cfg.MODEL.MASK_ON else None
            self.keypoint_inference(targets) if self.cfg.MODEL.KEYPOINT_ON else None
            self.parsing_inference(targets) if self.cfg.MODEL.PARSING_ON else None
            self.uv_inference(targets) if self.cfg.MODEL.UV_ON else None
        else:
            assert self.cfg.TEST.IMS_PER_GPU == 1, 'Instance batch test only supports IMS_PER_GPU == 1'
            batch_size = self.cfg.TEST.INSTANCES_PER_BATCH
            targets_list = instancebox_split(targets[0], batch_size)
            targets_list = [[targets] for targets in targets_list]
            if self.cfg.TEST.AFFINE_MODE == 'cv2' and len(inputs[0]) > 0:
                inputs_list = inputs[0].split(batch_size, dim=0)
                inputs_list = [[_] for _ in inputs_list]

            result_list = []
            for i, targets in enumerate(targets_list):
                if self.cfg.TEST.AFFINE_MODE == 'cv2' and len(inputs[0]) > 0:
                    inputs = inputs_list[i]
                self.conv_body_inference(inputs, targets)
                if len(self.features) == 0:
                    return self.result
                self.mask_inference(targets) if self.cfg.MODEL.MASK_ON else None
                self.keypoint_inference(targets) if self.cfg.MODEL.KEYPOINT_ON else None
                self.parsing_inference(targets) if self.cfg.MODEL.PARSING_ON else None
                self.uv_inference(targets) if self.cfg.MODEL.UV_ON else None
                result_list.append(self.result.copy())
            for task in self.result.keys():
                for key in self.result[task].keys():
                    if self.result[task][key] is not None:
                        self.result[task][key] = [torch.cat([result[task][key][0] for result in result_list])]

        return self.result

    def conv_body_inference(self, inputs, targets):
        _inputs = self.preprocess_inputs(inputs, targets)
        if _inputs is not None:
            features = [self.model.conv_body_net(_inputs)]
            if self.cfg.TEST.AUG.H_FLIP:
                inputs_hf = self.preprocess_inputs(inputs, targets, flip=True)
                features.append(self.model.conv_body_net(inputs_hf))
            for scale in self.cfg.TEST.AUG.SCALES:
                inputs_scl = self.preprocess_inputs(inputs, targets, scale=scale)
                features.append(self.model.conv_body_net(inputs_scl))
                if self.cfg.TEST.AUG.H_FLIP:
                    inputs_scl_hf = self.preprocess_inputs(inputs, targets, scale=scale, flip=True)
                    features.append(self.model.conv_body_net(inputs_scl_hf))

            self.features = features
        else:
            self.features = []

    def mask_inference(self, targets):
        labels = torch.cat([target.labels for target in targets if len(target.labels) > 0])
        mask_probs = []
        mask_iou_scores = []
        aug_idx = 0
        size = [self.cfg.TEST.SCALE[1], self.cfg.TEST.SCALE[0]]

        outputs = self.model.mask_net(self.features[aug_idx], labels)
        probs = outputs['probs']
        mask_probs.append(probs)
        mask_iou_scores.append(outputs['mask_iou_scores'])
        aug_idx += 1

        if self.cfg.TEST.AUG.H_FLIP:
            outputs_hf = self.model.mask_net(self.features[aug_idx], labels)
            probs_hf = outputs_hf['probs']
            probs_hf = torch.flip(probs_hf, dims=(3,))
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                probs_hf[:, :, :, 1:] = probs_hf[:, :, :, 0:-1]
            mask_probs.append(probs_hf)
            mask_iou_scores.append(outputs_hf['mask_iou_scores'])
            aug_idx += 1

        for scale in self.cfg.TEST.AUG.SCALES:
            outputs_scl = self.model.mask_net(self.features[aug_idx], labels)
            probs_scl = F.interpolate(outputs_scl['probs'], size=size, mode="bilinear", align_corners=False)
            mask_probs.append(probs_scl)
            mask_iou_scores.append(outputs_scl['mask_iou_scores'])
            aug_idx += 1

            if self.cfg.TEST.AUG.H_FLIP:
                outputs_scl_hf = self.model.mask_net(self.features[aug_idx], labels)
                probs_scl_hf = F.interpolate(outputs_scl_hf['probs'], size=size, mode="bilinear", align_corners=False)
                probs_scl_hf = torch.flip(probs_scl_hf, dims=(3,))
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                    probs_scl_hf[:, :, :, 1:] = probs_scl_hf[:, :, :, 0:-1]
                mask_probs.append(probs_scl_hf)
                mask_iou_scores.append(outputs_scl_hf['mask_iou_scores'])
                aug_idx += 1
        mask_iou_scores = torch.stack(mask_iou_scores, dim=0)
        mask_iou_scores = torch.mean(mask_iou_scores, dim=0)

        ims_bitmasks, mask_pixle_scores = get_mask_results(self.cfg, mask_probs, targets)
        boxes_per_image = [len(target) for target in targets]
        mask_iou_scores = mask_iou_scores.split(boxes_per_image, dim=0)
        mask_iou_scores = [_.cpu() for _ in mask_iou_scores]

        self.result['mask'] = dict(
            ims_bitmasks=ims_bitmasks, mask_iou_scores=mask_iou_scores, mask_pixle_scores=mask_pixle_scores
        )

    def keypoint_inference(self, targets):
        keypoint_probs = []
        kpt_iou_scores = []
        aug_idx = 0
        size = [self.cfg.KEYPOINT.PROB_SIZE[1], self.cfg.KEYPOINT.PROB_SIZE[0]]

        outputs = self.model.keypoint_net(self.features[aug_idx])
        probs = outputs['probs']
        keypoint_probs.append(probs)
        kpt_iou_scores.append(outputs['kpt_iou_scores'])
        aug_idx += 1

        if self.cfg.TEST.AUG.H_FLIP:
            outputs_hf = self.model.keypoint_net(self.features[aug_idx])
            probs_hf = outputs_hf['probs']
            probs_hf = flip_keypoints_prob(probs_hf)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                probs_hf[:, :, :, 1:] = probs_hf[:, :, :, 0:-1]
            keypoint_probs.append(probs_hf)
            kpt_iou_scores.append(outputs_hf['kpt_iou_scores'])
            aug_idx += 1

        for scale in self.cfg.TEST.AUG.SCALES:
            outputs_scl = self.model.keypoint_net(self.features[aug_idx])
            probs_scl = F.interpolate(outputs_scl['probs'], size=size, mode="bilinear", align_corners=False)
            keypoint_probs.append(probs_scl)
            kpt_iou_scores.append(outputs_scl['kpt_iou_scores'])
            aug_idx += 1

            if self.cfg.TEST.AUG.H_FLIP:
                outputs_scl_hf = self.model.keypoint_net(self.features[aug_idx])
                probs_scl_hf = F.interpolate(outputs_scl_hf['probs'], size=size, mode="bilinear", align_corners=False)
                probs_scl_hf = flip_keypoints_prob(probs_scl_hf)
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                    probs_scl_hf[:, :, :, 1:] = probs_scl_hf[:, :, :, 0:-1]
                keypoint_probs.append(probs_scl_hf)
                kpt_iou_scores.append(outputs_scl_hf['kpt_iou_scores'])
                aug_idx += 1
        kpt_iou_scores = torch.stack(kpt_iou_scores, dim=0)
        kpt_iou_scores = torch.mean(kpt_iou_scores, dim=0)
        keypoint_probs = torch.stack(keypoint_probs, dim=0)
        keypoint_probs = torch.mean(keypoint_probs, dim=0)

        ims_kpts, kpt_pixle_scores = get_keypoints_results(self.cfg, keypoint_probs, targets)
        boxes_per_image = [len(target) for target in targets]
        kpt_iou_scores = kpt_iou_scores.split(boxes_per_image, dim=0)
        kpt_iou_scores = [_.cpu() for _ in kpt_iou_scores]

        self.result['keypoints'] = dict(
            ims_kpts=ims_kpts, kpt_iou_scores=kpt_iou_scores, kpt_pixle_scores=kpt_pixle_scores
        )

    def parsing_inference(self, targets):
        flip_map = self.extra_fields['flip_map'] if 'flip_map' in self.extra_fields else ()
        parsing_probs = []
        parsing_iou_scores = []
        aug_idx = 0
        size = [self.cfg.TEST.SCALE[1], self.cfg.TEST.SCALE[0]]

        outputs = self.model.parsing_net(self.features[aug_idx])
        probs = outputs['probs']
        parsing_probs.append(probs)
        parsing_iou_scores.append(outputs['parsing_iou_scores'])
        aug_idx += 1

        if self.cfg.TEST.AUG.H_FLIP:
            outputs_hf = self.model.parsing_net(self.features[aug_idx])
            probs_hf = outputs_hf['probs']
            probs_hf = flip_parsing_prob(probs_hf, flip_map)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                probs_hf[:, :, :, 1:] = probs_hf[:, :, :, 0:-1]
            parsing_probs.append(probs_hf)
            parsing_iou_scores.append(outputs_hf['parsing_iou_scores'])
            aug_idx += 1

        for scale in self.cfg.TEST.AUG.SCALES:
            outputs_scl = self.model.parsing_net(self.features[aug_idx])
            probs_scl = F.interpolate(outputs_scl['probs'], size=size, mode="bilinear", align_corners=False)
            parsing_probs.append(probs_scl)
            parsing_iou_scores.append(outputs_scl['parsing_iou_scores'])
            aug_idx += 1

            if self.cfg.TEST.AUG.H_FLIP:
                outputs_scl_hf = self.model.parsing_net(self.features[aug_idx])
                probs_scl_hf = F.interpolate(outputs_scl_hf['probs'], size=size, mode="bilinear", align_corners=False)
                probs_scl_hf = flip_parsing_prob(probs_scl_hf, flip_map)
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if self.cfg.TEST.AUG.SHIFT_HEATMAP:
                    probs_scl_hf[:, :, :, 1:] = probs_scl_hf[:, :, :, 0:-1]
                parsing_probs.append(probs_scl_hf)
                parsing_iou_scores.append(outputs_scl_hf['parsing_iou_scores'])
                aug_idx += 1
        parsing_iou_scores = torch.stack(parsing_iou_scores, dim=0)
        parsing_iou_scores = torch.mean(parsing_iou_scores, dim=0)

        ims_parsings, parsing_instance_pixel_scores, parsing_part_pixel_scores = \
            get_parsing_results(self.cfg, parsing_probs, targets)
        boxes_per_image = [len(target) for target in targets]
        parsing_iou_scores = parsing_iou_scores.split(boxes_per_image, dim=0)
        parsing_iou_scores = [_.cpu() for _ in parsing_iou_scores]

        self.result['parsing'] = dict(
            ims_parsings=ims_parsings, parsing_iou_scores=parsing_iou_scores,
            parsing_instance_pixel_scores=parsing_instance_pixel_scores,
            parsing_part_pixel_scores=parsing_part_pixel_scores
        )

    def uv_inference(self, targets):
        uv_probs = [[], [], [], []]
        uv_iou_scores = []
        aug_idx = 0
        size = [self.cfg.TEST.SCALE[1], self.cfg.TEST.SCALE[0]]

        outputs = self.model.uv_net(self.features[aug_idx])
        add_uv_results(uv_probs, outputs['probs'])
        uv_iou_scores.append(outputs['uv_iou_scores'])
        aug_idx += 1

        if self.cfg.TEST.AUG.H_FLIP:
            outputs_hf = self.model.uv_net(self.features[aug_idx])
            probs_hf = flip_uv_prob(outputs_hf['probs'])
            add_uv_results(uv_probs, probs_hf)
            uv_iou_scores.append(outputs_hf['uv_iou_scores'])
            aug_idx += 1

        for scale in self.cfg.TEST.AUG.SCALES:
            outputs_scl = self.model.uv_net(self.features[aug_idx])
            probs_scl = [
                F.interpolate(prob, size=size, mode="bilinear", align_corners=False) for prob in outputs_scl['probs']
            ]
            add_uv_results(uv_probs, probs_scl)
            uv_iou_scores.append(outputs_scl['uv_iou_scores'])
            aug_idx += 1

            if self.cfg.TEST.AUG.H_FLIP:
                outputs_scl_hf = self.model.uv_net(self.features[aug_idx])
                probs_scl_hf = [
                    F.interpolate(prob, size=size, mode="bilinear", align_corners=False)
                    for prob in outputs_scl_hf['probs']
                ]
                probs_scl_hf = flip_uv_prob(probs_scl_hf)
                add_uv_results(uv_probs, probs_scl_hf)
                uv_iou_scores.append(outputs_scl_hf['uv_iou_scores'])
                aug_idx += 1

        _uv_probs = []
        for i in range(4):
            probs_ts = torch.stack(uv_probs[i], dim=0)
            _uv_probs.append(torch.mean(probs_ts, dim=0))
        uv_iou_scores = torch.stack(uv_iou_scores, dim=0)
        uv_iou_scores = torch.mean(uv_iou_scores, dim=0)

        ims_Index_UV, ims_U_uv, ims_V_uv, uv_pixel_scores = get_uv_results(self.cfg, _uv_probs, targets)
        boxes_per_image = [len(target) for target in targets]
        uv_iou_scores = uv_iou_scores.split(boxes_per_image, dim=0)
        uv_iou_scores = [_.cpu() for _ in uv_iou_scores]

        self.result['uv'] = dict(
            ims_Index_UV=ims_Index_UV, ims_U_uv=ims_U_uv, ims_V_uv=ims_V_uv,
            uv_iou_scores=uv_iou_scores, uv_pixel_scores=uv_pixel_scores
        )


class PreprocessInputs(object):
    def __init__(self, cfg):
        self.cfg = cfg
        pixel_stds = torch.from_numpy(np.array(cfg.PIXEL_STDS)).float()
        pixel_means = torch.from_numpy(np.array(cfg.PIXEL_MEANS)).float()
        self.pixel_stds = pixel_stds.view(1, 3, 1, 1).to(torch.device(self.cfg.DEVICE))
        self.pixel_means = pixel_means.view(1, 3, 1, 1).to(torch.device(self.cfg.DEVICE))

    def __call__(self, inputs, targets, scale=1.0, flip=False):
        if self.cfg.TEST.AFFINE_MODE == 'roi_align':
            inputs = [torch.from_numpy(_.transpose(2, 0, 1)).float() for _ in inputs]
            inputs = to_image_list(inputs, 0).to(torch.device(self.cfg.DEVICE))
            rois = []
            for i, target in enumerate(targets):
                box = target.bbox.clone()
                box[:, 2:4] *= scale
                batch_inds = torch.ones(len(box), device=box.device).to(dtype=box.dtype)[:, None] * i
                rois.append(torch.cat([batch_inds, box], dim=1))
            rois = torch.cat(rois).to(torch.device(self.cfg.DEVICE))
            if len(rois) == 0:
                return None
            inputs = (
                roi_align_rotated(inputs.tensors, rois, (self.cfg.TEST.SCALE[1], self.cfg.TEST.SCALE[0]), 1.0, 0, True)
            )
        elif self.cfg.TEST.AFFINE_MODE == 'cv2':
            assert scale == 1, 'cv2 only supports single scale'
            inputs = list(inputs)
            inputs = [_ for _ in inputs if len(_) > 0]
            if len(inputs) == 0:
                return None
            inputs = torch.cat(inputs, dim=0).to(torch.device(self.cfg.DEVICE)).float()
        inputs = (inputs / 255. - self.pixel_means) / self.pixel_stds
        if flip:
            inputs = inputs.flip(3)

        return inputs


def add_uv_results(all_results, results):
    for i in range(4):
        all_results[i].append(results[i])
