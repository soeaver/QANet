import torch

from pet.instance.modeling.uv_head import heads
from pet.instance.modeling.uv_head import outputs
from pet.instance.modeling.uv_head.loss import UV_loss
from pet.instance.modeling import registry
from pet.instance.core.config import cfg


class UV(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(UV, self).__init__()
        head = registry.UV_HEADS[cfg.UV.UV_HEAD]
        self.Head = head(dim_in)
        output = registry.UV_OUTPUTS[cfg.UV.UV_OUTPUT]
        self.Output = output(self.Head.dim_out, spatial_scale)

        self.loss_evaluator = UV_loss

    def forward(self, conv_features, targets=None):
        parsing_feat = self.Head(conv_features)
        output = self.Output(parsing_feat)

        if not self.training:
            return output, {}

        device = torch.device(cfg.DEVICE)
        UV_targets = targets['uv'].to(device)
        UV_masks = targets['uv_mask'].to(device)
        loss_seg_AnnIndex, loss_IndexUVPoints, loss_Upoints, loss_Vpoints = \
            self.loss_evaluator(output, UV_targets, UV_masks)

        loss_dict = dict(loss_Upoints=loss_Upoints, loss_Vpoints=loss_Vpoints,
                         loss_seg_Ann=loss_seg_AnnIndex, loss_IPoints=loss_IndexUVPoints)

        return None, loss_dict
