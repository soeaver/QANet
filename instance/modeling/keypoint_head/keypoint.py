import torch

from pet.instance.modeling.keypoint_head import heads
from pet.instance.modeling.keypoint_head import outputs
from pet.instance.modeling.keypoint_head.loss import keypoint_loss
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class Keypoint(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(Keypoint, self).__init__()
        head = registry.KEYPOINT_HEADS[cfg.KEYPOINT.KEYPOINT_HEAD]
        self.Head = head(dim_in)
        output = registry.KEYPOINT_OUTPUTS[cfg.KEYPOINT.KEYPOINT_OUTPUT]
        self.Output = output(self.Head.dim_out, spatial_scale)

        self.loss_evaluator = keypoint_loss

    def forward(self, conv_features, targets=None):
        keypoint_feat = self.Head(conv_features)
        output = self.Output(keypoint_feat)

        if not self.training:
            return output, {}

        device = torch.device(cfg.DEVICE)
        kp_targets = targets['keypoints'].to(device)
        target_weight = targets['keypoints_weight'].to(device) if cfg.KEYPOINT.USE_TARGET_WEIGHT else None
        loss_kp = self.loss_evaluator(output, kp_targets, target_weight)

        return None, dict(loss_kp=loss_kp)
