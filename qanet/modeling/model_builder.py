import torch
import torch.nn as nn

import qanet.modeling.backbone
import qanet.modeling.fpn
from qanet.modeling import registry
from qanet.modeling.keypoint_head.keypoint import Keypoint
from qanet.modeling.mask_head.mask import Mask
from qanet.modeling.parsing_head.parsing import Parsing
from qanet.modeling.uv_head.uv import UV


class Generalized_CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Backbone
        conv_body = registry.BACKBONES[self.cfg.BACKBONE.CONV_BODY]
        self.Conv_Body = conv_body(self.cfg)
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_in = self.Conv_Body.spatial_out

        # Feature Pyramid Networks
        if self.cfg.MODEL.FPN_ON:
            fpn_body = registry.FPN_BODY[self.cfg.FPN.BODY]
            self.Conv_Body_FPN = fpn_body(self.cfg, self.dim_in, self.spatial_in)
            self.dim_in = self.Conv_Body_FPN.dim_out
            self.spatial_in = self.Conv_Body_FPN.spatial_out
        else:
            self.dim_in = self.dim_in[-1:]
            self.spatial_in = self.spatial_in[-1:]

        if self.cfg.MODEL.MASK_ON:
            self.Mask = Mask(self.cfg, self.dim_in, self.spatial_in)

        if self.cfg.MODEL.KEYPOINT_ON:
            self.Keypoint = Keypoint(self.cfg, self.dim_in, self.spatial_in)

        if self.cfg.MODEL.PARSING_ON:
            self.Parsing = Parsing(self.cfg, self.dim_in, self.spatial_in)

        if self.cfg.MODEL.UV_ON:
            self.UV = UV(self.cfg, self.dim_in, self.spatial_in)

    def forward(self, x, targets=None):
        # Backbone
        conv_features = self.Conv_Body(x)

        # FPN
        if self.cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        results = []
        losses = {}
        if self.cfg.MODEL.MASK_ON:
            result_mask, loss_mask = self.Mask(conv_features, targets)
            results.append(result_mask)
            losses.update(loss_mask)
        if self.cfg.MODEL.KEYPOINT_ON:
            result_keypoint, loss_keypoint = self.Keypoint(conv_features, targets)
            results.append(result_keypoint)
            losses.update(loss_keypoint)
        if self.cfg.MODEL.PARSING_ON:
            result_parsing, loss_parsing = self.Parsing(conv_features, targets)
            results.append(result_parsing)
            losses.update(loss_parsing)
        if self.cfg.MODEL.UV_ON:
            result_uv, loss_uv = self.UV(conv_features, targets)
            results.append(result_uv)
            losses.update(loss_uv)

        if self.training:
            outputs = {'metrics': {}, 'losses': {}}
            outputs['losses'].update(losses)
            return outputs

        return results

    def conv_body_net(self, x):
        conv_features = self.Conv_Body(x)

        if self.cfg.MODEL.FPN_ON:
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
