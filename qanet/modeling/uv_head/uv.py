import torch

from qanet.modeling import registry
from qanet.modeling.uv_head import heads, outputs
from qanet.modeling.uv_head.loss import UV_loss_evaluator


class UV(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(UV, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in
        head = registry.UV_HEADS[cfg.UV.UV_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_in)
        output = registry.UV_OUTPUTS[cfg.UV.UV_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

        self.loss_evaluator = UV_loss_evaluator(cfg)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        uv_feat = self.Head(conv_features)
        logits = self.Output(uv_feat)

        loss_seg_AnnIndex, loss_IndexUVPoints, loss_Upoints, loss_Vpoints = \
            self.loss_evaluator(logits, targets['uv'], targets['uv_mask'])
        loss_dict = dict(loss_Upoints=loss_Upoints, loss_Vpoints=loss_Vpoints,
                         loss_seg_Ann=loss_seg_AnnIndex, loss_IPoints=loss_IndexUVPoints)

        return None, loss_dict

    def _forward_test(self, conv_features, targets=None):
        uv_feat = self.Head(conv_features)
        uv_logits = self.Output(uv_feat)
        # uv_logits = logits[-1]
        return dict(probs=uv_logits, uv_iou_scores=torch.ones(uv_logits[0].size()[0], dtype=torch.float32,
                                                              device=uv_logits[0].device)), {}
