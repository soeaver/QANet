import torch
import torch.nn as nn

import instance.modeling.backbone
import instance.modeling.fpn
from instance.modeling.mask_head.mask import Mask
from instance.modeling.keypoint_head.keypoint import Keypoint
from instance.modeling.parsing_head.parsing import Parsing
from instance.modeling.uv_head.uv import UV
from instance.modeling.qanet_head.qanet import QANet
from instance.core.config import cfg
from instance.modeling import registry


class Generalized_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        conv_body = registry.BACKBONES[cfg.BACKBONE.CONV_BODY]
        self.Conv_Body = conv_body()
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_scale = self.Conv_Body.spatial_scale

        # Feature Pyramid Networks
        if cfg.MODEL.FPN_ON:
            fpn_body = registry.FPN_BODY[cfg.FPN.BODY]
            self.Conv_Body_FPN = fpn_body(self.dim_in, self.spatial_scale)
            self.dim_in = self.Conv_Body_FPN.dim_out
            self.spatial_scale = self.Conv_Body_FPN.spatial_scale
        else:
            self.dim_in = self.dim_in[-1:]
            self.spatial_scale = self.spatial_scale[-1:]

        if cfg.MODEL.MASK_ON:
            self.Mask = Mask(self.dim_in, self.spatial_scale)

        if cfg.MODEL.KEYPOINT_ON:
            self.Keypoint = Keypoint(self.dim_in, self.spatial_scale)

        if cfg.MODEL.PARSING_ON:
            self.Parsing = Parsing(self.dim_in, self.spatial_scale)

        if cfg.MODEL.UV_ON:
            self.UV = UV(self.dim_in, self.spatial_scale)

        if cfg.MODEL.QANET_ON:
            self.QANet = QANet(self.dim_in, self.spatial_scale)

    def forward(self, x, targets=None):
        # Backbone
        conv_features = self.Conv_Body(x)

        # FPN
        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        losses = {}
        if cfg.MODEL.MASK_ON:
            result_mask, loss_mask = self.Mask(conv_features, targets)
            losses.update(loss_mask)
        if cfg.MODEL.KEYPOINT_ON:
            result_keypoint, loss_keypoint = self.Keypoint(conv_features, targets)
            losses.update(loss_keypoint)
        if cfg.MODEL.PARSING_ON:
            result_parsing, loss_parsing = self.Parsing(conv_features, targets)
            losses.update(loss_parsing)
        if cfg.MODEL.UV_ON:
            result_uv, loss_uv = self.UV(conv_features, targets)
            losses.update(loss_uv)
        if cfg.MODEL.QANET_ON:
            result_qanet, loss_qanet = self.QANet(conv_features, targets)
            losses.update(loss_qanet)

        if self.training:
            outputs = {'metrics': {}, 'losses': {}}
            outputs['losses'].update(losses)
            return outputs

        return None

    def conv_body_net(self, x):
        conv_features = self.Conv_Body(x)

        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]
        return conv_features

    def mask_net(self, conv_features, targets=None):
        result_mask, loss_mask = self.Mask(conv_features, targets)
        return result_mask

    def keypoint_net(self, conv_features, targets=None):
        result_keypoint, loss_keypoint = self.Keypoint(conv_features, targets)
        return result_keypoint

    def parsing_net(self, conv_features, targets=None):
        result_parsing, loss_parsing = self.Parsing(conv_features, targets)
        return result_parsing

    def uv_net(self, conv_features, targets=None):
        result_uv, loss_uv = self.UV(conv_features, targets)
        return result_uv

    def qanet_net(self, conv_features, targets=None):
        result_qanet, loss_qanet = self.QANet(conv_features, targets)
        return result_qanet
