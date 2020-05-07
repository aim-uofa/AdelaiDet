from typing import Dict
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform


BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_basis_module(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.BLENDMASK.VISUALIZE
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        tower = []
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses
