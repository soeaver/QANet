import torch

from qanet.modeling import registry
from qanet.modeling.keypoint_head import heads, outputs
from qanet.modeling.keypoint_head.loss import keypoint_loss_evaluator


class Keypoint(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(Keypoint, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in
        self.use_target_weight = cfg.KEYPOINT.USE_TARGET_WEIGHT

        head = registry.KEYPOINT_HEADS[cfg.KEYPOINT.KEYPOINT_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)
        output = registry.KEYPOINT_OUTPUTS[cfg.KEYPOINT.KEYPOINT_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

        self.loss_evaluator = keypoint_loss_evaluator(cfg)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        keypoint_feat = self.Head(conv_features)
        logits = self.Output(keypoint_feat)

        loss_kp = self.loss_evaluator(
            logits, targets['keypoints'],
            targets['keypoints_weight'] if self.use_target_weight else None
        )
        return None, dict(loss_kp=loss_kp)

    def _forward_test(self, conv_features, targets=None):
        keypoint_feat = self.Head(conv_features)
        logits = self.Output(keypoint_feat)
        kpt_logits = logits[-1]

        return dict(probs=kpt_logits, kpt_iou_scores=torch.ones(kpt_logits.size()[0], dtype=torch.float32,
                                                            device=kpt_logits.device)), {}
