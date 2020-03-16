import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.ops import l2_loss
from instance.core.config import cfg
from utils.net import make_conv, make_fc


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


class quality_head(nn.Module):
    """
    quality head feature extractor.
    """

    def __init__(self, dim_in):
        super(quality_head, self).__init__()

        self.dim_in = dim_in + cfg.QANET.NUM_PARSING
        num_stacked_convs = 2
        conv_dim = 64
        mlp_dim = 256
        use_bn = False
        use_gn = False

        convx = []
        for _ in range(num_stacked_convs):
            layer_stride = 2 if _ == 0 else 1
            convx.append(
                make_conv(
                    self.dim_in, conv_dim, kernel=3, stride=layer_stride, use_bn=use_bn, use_gn=use_gn, use_relu=True
                )
            )
            self.dim_in = conv_dim
        self.convx = nn.Sequential(*convx)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        fcx = []
        for _ in range(2):
            fcx.append(
                make_conv(
                    self.dim_in, mlp_dim, kernel=1, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True
                )
            )
            self.dim_in = mlp_dim
        self.fcx = nn.Sequential(*fcx)

        self.dim_out = mlp_dim

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, parsing_logits):
        x = torch.cat((x, parsing_logits), 1)
        x = self.convx(x)

        x = self.avgpool(x)
        x = self.fcx(x)

        return x


class QALossComputation(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale, device):
        super(QALossComputation, self).__init__()
        self.dim_in = dim_in
        self.spatial_scale = spatial_scale[0]
        self.device = torch.device(device)
        self.num_parsing = cfg.QANET.NUM_PARSING

        self.classify = nn.Conv2d(dim_in, self.num_parsing, kernel_size=1, stride=1, padding=0)
        self.quality = quality_head(dim_in)
        self.regressor = nn.Linear(256, 1)

        enhanced_conv = []
        for _ in range(2):
            enhanced_conv.append(
                make_conv(
                    self.dim_in, self.dim_in, kernel=3, stride=1, use_bn=True, use_gn=False, use_relu=True
                )
            )
        self.enhanced_conv = nn.Sequential(*enhanced_conv)
        self.enhanced_classify = nn.Conv2d(self.dim_in, self.num_parsing, kernel_size=1, stride=1, padding=0)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_quality_targets(self, p_logites, p_targets):
        pred_parsings_np = p_logites.detach().argmax(dim=1).cpu().numpy()
        parsing_targets_np = p_targets.cpu().numpy()

        N = parsing_targets_np.shape[0]
        quality_targets = np.zeros(N, dtype=np.float)
        for _ in range(N):
            parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], cfg.PARSING.NUM_PARSING)
            quality_targets[_] = np.nanmean(parsing_iou)
        quality_targets = torch.from_numpy(quality_targets).to(self.device, dtype=torch.float)
        return quality_targets

    def forward(self, parsing_feat, parsing_targets=None, is_train=True):
        parsing_logits = self.classify(parsing_feat)

        quality_feat = self.quality(parsing_feat, parsing_logits)
        quality_logits = self.regressor(quality_feat.view(quality_feat.size(0), -1))

        enhanced_feat = F.sigmoid(quality_feat) * parsing_feat
        enhanced_feat = self.enhanced_conv(enhanced_feat)
        enhanced_parsing_logits = self.enhanced_classify(enhanced_feat)

        up_scale = int(1 / self.spatial_scale)
        if up_scale > 1:
            parsing_logits = F.interpolate(parsing_logits, scale_factor=up_scale, mode="bilinear", align_corners=False)
            enhanced_parsing_logits = F.interpolate(enhanced_parsing_logits, scale_factor=up_scale, mode="bilinear",
                                                    align_corners=False)

        if not is_train:
            return F.softmax(enhanced_parsing_logits, dim=1), quality_logits

        # targets
        quality_targets = self.generate_quality_targets(parsing_logits, parsing_targets)
        quality_targets = quality_targets.detach()
        parsing_targets = parsing_targets.to(self.device)

        # losses
        parsing_loss = F.cross_entropy(parsing_logits, parsing_targets)
        parsing_loss *= cfg.QANET.PARSING_LOSS_WEIGHT
        quality_loss = l2_loss(quality_logits[:, 0], quality_targets)
        quality_loss *= cfg.QANET.QUALITY_LOSS_WEIGHT
        enhanced_parsing_loss = F.cross_entropy(enhanced_parsing_logits, parsing_targets)
        enhanced_parsing_loss *= cfg.QANET.PARSING_LOSS_WEIGHT

        return parsing_loss + enhanced_parsing_loss, quality_loss

    # def forward(self, parsing_feat, parsing_targets=None, is_train=True):
    #     parsing_logits = self.classify(parsing_feat)
    #
    #     quality_feat = self.quality(parsing_feat, parsing_logits)
    #     quality_logits = self.regressor(quality_feat)
    #
    #     up_scale = int(1 / self.spatial_scale)
    #     if up_scale > 1:
    #         parsing_logits = F.interpolate(parsing_logits, scale_factor=up_scale, mode="bilinear", align_corners=False)
    #
    #     if not is_train:
    #         return F.softmax(parsing_logits, dim=1), quality_logits
    #
    #     # targets
    #     quality_targets = self.generate_quality_targets(parsing_logits, parsing_targets)
    #     quality_targets = quality_targets.detach()
    #     parsing_targets = parsing_targets.to(self.device)
    #
    #     # losses
    #     parsing_loss = F.cross_entropy(parsing_logits, parsing_targets)
    #     parsing_loss *= cfg.QANET.PARSING_LOSS_WEIGHT
    #     quality_loss = l2_loss(quality_logits[:, 0], quality_targets)
    #     quality_loss *= cfg.QANET.QUALITY_LOSS_WEIGHT
    #
    #     return parsing_loss, quality_loss


def qanet_loss_evaluator(dim_in, spatial_scale):
    loss_evaluator = QALossComputation(
        dim_in, spatial_scale, cfg.DEVICE
    )
    return loss_evaluator
