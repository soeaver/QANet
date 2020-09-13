import torch

from instance.modeling.uv_head import heads
from instance.modeling.uv_head import outputs
from instance.modeling.uv_head.loss import UV_loss_evaluator
from instance.modeling import registry


class UV(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(UV, self).__init__()
        self.spatial_scale = spatial_scale

        head = registry.UV_HEADS[cfg.UV.UV_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.UV_OUTPUTS[cfg.UV.UV_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_scale)

        self.loss_evaluator = UV_loss_evaluator(cfg)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        uv_feat = self.Head(conv_features)
        output = self.Output(uv_feat)

        loss_seg_AnnIndex, loss_IndexUVPoints, loss_Upoints, loss_Vpoints = \
            self.loss_evaluator(output, targets['uv'], targets['uv_mask'])
        loss_dict = dict(loss_Upoints=loss_Upoints, loss_Vpoints=loss_Vpoints,
                         loss_seg_Ann=loss_seg_AnnIndex, loss_IPoints=loss_IndexUVPoints)

        return None, loss_dict

    def _forward_test(self, conv_features, targets=None):
        uv_feat = self.Head(conv_features)
        output = self.Output(uv_feat)

        return dict(probs=output, uv_iou_scores=torch.ones(output[0].size()[0], dtype=torch.float32,
                                                            device=output[0].device)), {}