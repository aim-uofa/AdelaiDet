from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import FPN, build_resnet_backbone
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .resnet_lpf import build_resnet_lpf_backbone
from .resnet_interval import build_resnet_interval_backbone
from .mobilenet import build_mnv2_backbone


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features="res5", norm="", activation=False):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.use_relu = activation
        use_bias = norm == ""
        self.p6 = Conv2d(in_channels, out_channels, 3, 2, 1, bias=use_bias, norm=get_norm(norm, out_channels))
        self.p7 = Conv2d(out_channels, out_channels, 3, 2, 1, bias=use_bias, norm=get_norm(norm, out_channels))
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        if self.use_relu:
            p6 = F.relu_(p6)
            p7 = self.p7(p6)
        else:
            p7 = self.p7(F.relu(p6))
        if self.use_relu:
            p7 = F.relu_(p7)
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5", norm="", activation=False):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.use_relu = activation
        use_bias = norm == ""
        self.p6 = Conv2d(in_channels, out_channels, 3, 2, 1, bias=use_bias, norm=get_norm(norm, out_channels))
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        if self.use_relu:
            p6 = F.relu_(p6)
        return [p6]


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.BACKBONE.ANTI_ALIAS:
        bottom_up = build_resnet_lpf_backbone(cfg, input_shape)
    elif cfg.MODEL.RESNETS.DEFORM_INTERVAL > 1:
        bottom_up = build_resnet_interval_backbone(cfg, input_shape)
    elif cfg.MODEL.MOBILENET:
        bottom_up = build_mnv2_backbone(cfg, input_shape)
    else:
        bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    norm = cfg.MODEL.FPN.NORM
    activation = cfg.MODEL.FPN.USE_RELU
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5", norm=norm, activation=activation)
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5", norm=norm, activation=activation)
    elif top_levels == 0:
        top_block = None
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        activation=activation,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
