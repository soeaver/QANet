import torch
import torch.nn.functional as F

from instance.modeling.parsing_head import heads
from instance.modeling.parsing_head import outputs
from instance.modeling.parsing_head.loss import parsing_loss_evaluator
from instance.modeling.parsing_head.parsingiou.parsingiou import ParsingIoU
from instance.modeling import registry
from instance.core.config import cfg


class Parsing(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(Parsing, self).__init__()
        head = registry.PARSING_HEADS[cfg.PARSING.PARSING_HEAD]
        self.Head = head(dim_in)
        output = registry.PARSING_OUTPUTS[cfg.PARSING.PARSING_OUTPUT]
        self.Output = output(self.Head.dim_out, spatial_scale)

        self.loss_evaluator = parsing_loss_evaluator()

        if cfg.PARSING.PARSINGIOU_ON:
            self.ParsingIoU = ParsingIoU(self.Head.dim_out, spatial_scale)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features)

    def _forward_train(self, conv_features, targets=None):
        parsing_feat = self.Head(conv_features)
        parsing_logits = self.Output(parsing_feat)

        if cfg.PARSING.PARSINGIOU_ON:
            loss_parsing, parsingiou_targets = self.loss_evaluator(parsing_logits, targets['parsing'])
            loss_parsingiou, _ = self.ParsingIoU(parsing_feat, parsing_logits, parsingiou_targets)
            return None, dict(loss_parsing=loss_parsing, loss_parsingiou=loss_parsingiou)
        else:
            loss_parsing = self.loss_evaluator(parsing_logits, targets['parsing'])
            return None, dict(loss_parsing=loss_parsing)

    def _forward_test(self, conv_features):
        parsing_feat = self.Head(conv_features)
        parsing_logits = self.Output(parsing_feat)

        output = F.softmax(parsing_logits, dim=1)

        if cfg.PARSING.PARSINGIOU_ON:
            _, parsingiou = self.ParsingIoU(parsing_feat, parsing_logits, None)
            return dict(parsings=output, parsing_scores=parsingiou.squeeze()), {}
        else:
            return dict(parsings=output, parsing_scores=None), {}
