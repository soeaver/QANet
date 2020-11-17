import torch
import torch.nn.functional as F

from qanet.modeling import registry
from qanet.modeling.parsing_head import heads, outputs
from qanet.modeling.parsing_head.loss import parsing_loss_evaluator
from qanet.modeling.parsing_head.parsingiou.parsingiou import ParsingIoU


class Parsing(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(Parsing, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in
        self.parsingiou_on = cfg.PARSING.PARSINGIOU_ON

        head = registry.PARSING_HEADS[cfg.PARSING.PARSING_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)
        output = registry.PARSING_OUTPUTS[cfg.PARSING.PARSING_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.loss_evaluator = parsing_loss_evaluator(cfg)

        if self.parsingiou_on:
            self.ParsingIoU = ParsingIoU(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.dim_out = self.Output.dim_out
        self.spatial_out = self.Output.spatial_out

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features)

    def _forward_train(self, conv_features, targets=None):
        losses = dict()

        x = self.Head(conv_features)
        logits = self.Output(x)

        loss_parsing, parsingiou_targets = self.loss_evaluator(logits, targets['parsing'])
        losses.update(dict(loss_parsing=loss_parsing))

        if self.parsingiou_on:
            loss_parsingiou, _ = self.ParsingIoU(x, parsingiou_targets)
            losses.update(dict(loss_parsingiou=loss_parsingiou))

        return None, losses

    def _forward_test(self, conv_features):
        x = self.Head(conv_features)
        logits = self.Output(x)

        output = F.softmax(logits[-1], dim=1)
        results = dict(
            probs=output,
            parsing_iou_scores=torch.ones(output.size()[0], dtype=torch.float32, device=output.device)
        )

        if self.parsingiou_on:
            _, parsingiou = self.ParsingIoU(x, None)
            results.update(dict(parsing_iou_scores=parsingiou.squeeze(1)))

        return results, {}
