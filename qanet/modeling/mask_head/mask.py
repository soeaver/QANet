import torch

from qanet.modeling import registry
from qanet.modeling.mask_head import heads, outputs
from qanet.modeling.mask_head.loss import mask_loss_evaluator
from qanet.modeling.mask_head.maskiou.maskiou import MaskIoU


class Mask(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(Mask, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in
        self.maskiou_on = cfg.MASK.MASKIOU_ON

        head = registry.MASK_HEADS[cfg.MASK.MASK_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_in)
        output = registry.MASK_OUTPUTS[cfg.MASK.MASK_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.loss_evaluator = mask_loss_evaluator(cfg)

        if self.maskiou_on:
            self.MaskIoU = MaskIoU(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        losses = dict()

        x = self.Head(conv_features)
        logits = self.Output(x, targets['labels'])

        loss_mask, maskiou_targets = self.loss_evaluator(logits, targets['mask'])
        losses.update(dict(loss_mask=loss_mask))

        if self.maskiou_on:
            loss_maskiou, _ = self.MaskIoU(x, targets['labels'], maskiou_targets)
            losses.update(dict(loss_maskiou=loss_maskiou))

        return None, losses

    def _forward_test(self, conv_features, targets=None):
        x = self.Head(conv_features)
        logits = self.Output(x, targets)

        output = logits[-1].sigmoid()
        results = dict(
            probs=output,
            mask_iou_scores=torch.ones(output.size()[0], dtype=torch.float32, device=output.device)
        )

        if self.maskiou_on:
            _, maskiou = self.MaskIoU(x, targets, None)
            results.update(dict(mask_iou_scores=maskiou))

        return results, {}
