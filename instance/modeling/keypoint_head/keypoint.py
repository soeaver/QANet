import torch

from instance.modeling.keypoint_head import heads
from instance.modeling.keypoint_head import outputs
from instance.modeling.keypoint_head.loss import keypoint_loss_evaluator
from instance.modeling import registry


class Keypoint(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(Keypoint, self).__init__()
        self.spatial_scale = spatial_scale
        self.use_target_weight = cfg.KEYPOINT.USE_TARGET_WEIGHT

        head = registry.KEYPOINT_HEADS[cfg.KEYPOINT.KEYPOINT_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.KEYPOINT_OUTPUTS[cfg.KEYPOINT.KEYPOINT_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_scale)

        self.loss_evaluator = keypoint_loss_evaluator(cfg)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        keypoint_feat = self.Head(conv_features)
        output = self.Output(keypoint_feat)

        loss_kp = self.loss_evaluator(
            output, targets['keypoints'],
            targets['keypoints_weight'] if self.use_target_weight else None
        )
        return None, dict(loss_kp=loss_kp)

    def _forward_test(self, conv_features, targets=None):
        keypoint_feat = self.Head(conv_features)
        output = self.Output(keypoint_feat)

        return dict(probs=output, kpt_iou_scores=torch.ones(output.size()[0], dtype=torch.float32,
                                                            device=output.device)), {}
