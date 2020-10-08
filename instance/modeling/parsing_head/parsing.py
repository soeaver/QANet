import torch
import torch.nn.functional as F

from instance.modeling.parsing_head import heads
from instance.modeling.parsing_head import outputs
from instance.modeling.parsing_head.loss import parsing_loss_evaluator
from instance.modeling.parsing_head.parsingiou.parsingiou import ParsingIoU
from instance.modeling.parsing_head.quality.quality import Quality
from instance.modeling import registry


class Parsing(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_scale):
        super(Parsing, self).__init__()
        self.spatial_scale = spatial_scale
        self.parsingiou_on = cfg.PARSING.PARSINGIOU_ON
        self.quality_on = cfg.PARSING.QUALITY_ON

        if self.quality_on:
            self.Quality = Quality(cfg, dim_in, self.spatial_scale)
            dim_in = self.Quality.dim_out

        head = registry.PARSING_HEADS[cfg.PARSING.PARSING_HEAD]
        self.Head = head(cfg, dim_in, self.spatial_scale)
        output = registry.PARSING_OUTPUTS[cfg.PARSING.PARSING_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_scale)

        self.loss_evaluator = parsing_loss_evaluator(cfg)

        if self.parsingiou_on:
            self.ParsingIoU = ParsingIoU(cfg, self.Head.dim_out, self.Head.spatial_scale)

    def forward(self, conv_features, targets=None):
        if self.training:
            return self._forward_train(conv_features, targets)
        else:
            return self._forward_test(conv_features, targets)

    def _forward_train(self, conv_features, targets=None):
        losses = dict()

        if self.quality_on:
            loss_quality, conv_features = self.Quality(conv_features, targets['parsing'])
            losses.update(loss_quality)

        parsing_feat = self.Head(conv_features)
        parsing_logits = self.Output(parsing_feat)

        loss_parsing, parsingiou_targets = self.loss_evaluator(parsing_logits, targets['parsing'])
        losses.update(dict(loss_parsing=loss_parsing))

        if self.parsingiou_on:
            loss_parsingiou, _ = self.ParsingIoU(parsing_feat, parsingiou_targets)
            losses.update(dict(loss_parsingiou=loss_parsingiou))

        return None, losses

    def _forward_test(self, conv_features, targets=None):
        if self.quality_on:
            _, conv_features = self.Quality(conv_features, None)

        parsing_feat = self.Head(conv_features)
        parsing_logits = self.Output(parsing_feat)

        output = F.softmax(parsing_logits, dim=1)
        results = dict(
            probs=output,
            parsing_iou_scores=torch.ones(output.size()[0], dtype=torch.float32, device=output.device)
        )

        if self.parsingiou_on:
            _, parsingiou = self.ParsingIoU(parsing_feat, None)
            results.update(dict(parsing_iou_scorestorch.mean(parsingiou, dim=1)))

        return results, {}
