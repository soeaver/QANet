import torch

from instance.modeling.mask_head import heads
from instance.modeling.mask_head import outputs
from instance.modeling.mask_head.loss import mask_loss_evaluator
from instance.modeling.mask_head.maskiou.maskiou import MaskIoU
from instance.modeling import registry


class Mask(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(Mask, self).__init__()
        self.spatial_scale = spatial_scale
        self.maskiou_on = cfg.MASK.MASKIOU_ON

        head = registry.MASK_HEADS[cfg.MASK.MASK_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.MASK_OUTPUTS[cfg.MASK.MASK_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_scale)

        self.loss_evaluator = mask_loss_evaluator(cfg)

        if self.maskiou_on:
            self.MaskIoU = MaskIoU(cfg, self.Head.dim_out, self.Head.spatial_scale)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        mask_feat = self.Head(conv_features)
        mask_logits = self.Output(mask_feat, targets['labels'])

        if self.maskiou_on:
            loss_mask, maskiou_targets = self.loss_evaluator(mask_logits, targets['mask'])
            loss_maskiou, _ = self.MaskIoU(mask_feat, mask_logits, targets['labels'], maskiou_targets)
            return None, dict(loss_mask=loss_mask, loss_maskiou=loss_maskiou)
        else:
            loss_mask = self.loss_evaluator(mask_logits, targets['mask'])
            return None, dict(loss_mask=loss_mask)

    def _forward_test(self, conv_features, targets=None):
        mask_feat = self.Head(conv_features)
        mask_logits = self.Output(mask_feat, targets)

        output = mask_logits.sigmoid()

        if self.maskiou_on:
            _, maskiou = self.MaskIoU(mask_feat, mask_logits, targets, None)
            return dict(probs=output, mask_iou_scores=maskiou), {}
        else:
            return dict(probs=output, mask_iou_scores=torch.ones(output.size()[0], dtype=torch.float32,
                                                                 device=output.device)), {}
