import torch
import torch.nn.functional as F

from instance.modeling.qanet_head import heads
from instance.modeling.qanet_head.loss import qanet_loss_evaluator
from instance.modeling import registry
from instance.core.config import cfg


class QANet(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(QANet, self).__init__()
        head = registry.QANET_HEADS[cfg.QANET.QANET_HEAD]
        self.Head = head(dim_in)

        self.loss_evaluator = qanet_loss_evaluator(self.Head.dim_out, spatial_scale)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features)

    def _forward_train(self, conv_features, targets=None):
        parsing_feat = self.Head(conv_features)

        loss_parsing, loss_quality = self.loss_evaluator(parsing_feat, targets['parsing'], is_train=True)
        return None, dict(loss_parsing=loss_parsing, loss_quality=loss_quality)

    def _forward_test(self, conv_features):
        parsing_feat = self.Head(conv_features)

        parsings, quality = self.loss_evaluator(parsing_feat, None, is_train=False)
        return dict(parsings=parsings, parsing_scores=quality.squeeze()), {}
