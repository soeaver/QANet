import torch
from torch import nn
from torch.nn import functional as F

from lib.layers import make_conv, make_norm, make_fc, make_act

from instance.modeling.parsing_head.quality import heads, outputs
from instance.modeling.parsing_head.quality.loss import quality_loss_evaluator
from instance.modeling import registry


class QualityEncoder(torch.nn.Module):
    def __init__(self, cfg, dim_in, dropout=0.05):
        super(QualityEncoder, self).__init__()
        self.dim_in = dim_in[-1]

        q_dim = cfg.PARSING.QUALITY.QUALITY_DIM  # 256
        output_dim = cfg.PARSING.QUALITY.OUTPUT_DIM  # 512
        num_parsing = cfg.PARSING.NUM_PARSING  # 20
        norm = cfg.PARSING.QUALITY.NORM
        self.q_dim = q_dim

        self.feat_conv3x3 = make_conv(self.dim_in, q_dim, 3, 1, 1, norm=make_norm(q_dim, norm=norm), act=make_act())
        self.prob_conv3x3 = make_conv(num_parsing, q_dim, 3, 1, 1, norm=make_norm(q_dim, norm=norm), act=make_act())
        self.iou_conv1x1 = make_conv(1, q_dim, 1, 1, 0, norm=make_norm(q_dim, norm=norm), act=make_act())

        self.prob_quality = make_conv(q_dim, q_dim, 3, 1, 1, norm=make_norm(q_dim, norm=norm), act=make_act())
        self.iou_quality = make_conv(1, q_dim, 3, 1, 1, norm=make_norm(q_dim, norm=norm), act=make_act())

        self.conv_out = nn.Sequential(
            make_conv(q_dim * 2, output_dim, 3, 1, 1, norm=make_norm(output_dim, norm=norm), act=make_act()),
            # nn.Dropout2d(dropout)
        )

        self.dim_out = [output_dim]

    def forward(self, features, parsing_logits, iou_pred):
        """
        Arguments:
            features (Tensor): feature-maps from possibly several levels
            parsing_logits (Tensor): parsing logits
            iou_pred (Tensor, optional): iou prediction.

        Returns:
            encoded_features (Tensor): encoded conv features
        """
        feat = features[-1]
        b, c, h, w = feat.size()
        # [b, c, h, w] -> [b, q_dim, h, w]
        feat = self.feat_conv3x3(feat)
        # [b, q_dim, h, w] -> [b, q_dim, hw] -> [b, hw, q_dim]
        feat_key = feat.view(b, self.q_dim, -1).permute(0, 2, 1)

        # [b, num_parsing, h, w] -> [b, q_dim, h, w] -> [b, q_dim, hw]
        prob_query = self.prob_conv3x3(parsing_logits).view(b, self.q_dim, -1)

        # [b, 1, 1, 1] -> [b, q_dim, 1, 1] -> [b, q_dim, 1]
        iou_query = self.iou_conv1x1(iou_pred.view(b, 1, 1, 1)).view(b, self.q_dim, -1)

        # [b, q_dim, hw] * [b, hw, q_dim] -> [b, q_dim, q_dim] -> [b, q_dim, q_dim, 1]
        prob_quality = torch.matmul(prob_query, feat_key).unsqueeze(3)
        # [b, q_dim, q_dim, 1] -> [b, q_dim, q_dim, 1] -> [b, q_dim, q_dim]
        prob_quality = self.prob_quality(prob_quality).view(b, self.q_dim, -1)
        prob_quality = F.softmax(prob_quality, dim=-1)

        # [b, hw, q_dim] * [b, q_dim, 1] -> [b, hw, 1] -> [b, 1, hw] -> [b, 1, h, w]
        iou_quality = torch.matmul(feat_key, iou_query).permute(0, 2, 1).view(b, 1, h, w)
        # [b, 1, h, w] -> [b, q_dim, h, w] -> [b, q_dim, hw]
        iou_quality = self.iou_quality(iou_quality).view(b, self.q_dim, -1)

        # [b, q_dim, q_dim] * [b, q_dim, hw] -> [b, q_dim, hw] -> [b, q_dim, h, w]
        quality_feat = torch.matmul(prob_quality, iou_quality).view(b, self.q_dim, h, w)

        quality_feat = torch.cat((feat, quality_feat), 1)
        quality_feat = self.conv_out(quality_feat)

        return [quality_feat]


class Quality(torch.nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        super(Quality, self).__init__()
        self.dim_in = dim_in
        self.spatial_in = spatial_in

        head = registry.QUALITY_HEADS[cfg.PARSING.QUALITY.QUALITY_HEAD]
        self.Head = head(cfg, self.dim_in, self.spatial_in)
        output = registry.QUALITY_OUTPUTS[cfg.PARSING.QUALITY.QUALITY_OUTPUT]
        self.Output = output(cfg, self.Head.dim_out, self.Head.spatial_out)

        self.QualityEncoder = QualityEncoder(cfg, self.dim_in)

        self.loss_evaluator = quality_loss_evaluator(cfg)

        self.dim_out = self.QualityEncoder.dim_out
        self.spatial_out = spatial_in

    def forward(self, features, parsing_targets=None):
        """
        Arguments:
            features (Tensor): feature-maps from possibly several levels
            parsing_targets (Tensor, optional): the ground-truth parsing targets.

        Returns:
            losses (Tensor): During training, returns the losses for the
                head. During testing, returns an empty dict.
            encoded_features (Tensor): during training, returns None. During testing, the predicted parsingiou.
        """
        x = self.Head(features)
        parsing_logits, iou_pred = self.Output(x)

        quality_features = self.QualityEncoder(features, parsing_logits, iou_pred)

        if self.training:
            return self._forward_train(quality_features, parsing_logits, iou_pred, parsing_targets)
        else:
            return self._forward_test(quality_features)

    def _forward_train(self, quality_features, parsing_logits, iou_pred, parsing_targets=None):
        loss_quality = self.loss_evaluator(parsing_logits, iou_pred, parsing_targets)
        return loss_quality, quality_features

    def _forward_test(self, quality_features):
        return {}, quality_features
