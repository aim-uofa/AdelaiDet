from typing import Dict
from torch import nn
from torch.nn import functional as F
import torch

from detectron2.layers import ShapeSpec
from .utils import aligned_bilinear, compute_loss

from adet.layers import conv_with_kaiming_uniform
from detectron2.structures import ImageList
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.utils.comm import get_world_size
from adet.utils.comm import reduce_sum
import math
from detectron2.layers import ConvTranspose2d
from detectron2.layers.batch_norm import get_norm


class basis_module(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.FCPOSE.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.FCPOSE.BASIS_MODULE.CONVS_DIM
        self.device       = torch.device(cfg.MODEL.DEVICE)
        self.in_features  = ["p3", "p4", "p5"]
        self.loss_on      = True
        norm              = cfg.MODEL.FCPOSE.BASIS_MODULE.BN_TYPE #"SyncBN"
        num_convs         = 3
        self.visualize    = False
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
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim+(2*17), 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.FCPOSE.BASIS_MODULE.COMMON_STRIDE
            self.num_classes          = cfg.MODEL.FCPOSE.BASIS_MODULE.NUM_CLASSES
            self.heatmap_loss_weight = cfg.MODEL.FCPOSE.BASIS_MODULE.LOSS_WEIGHT
            # self.focal_loss_alpha = cfg.MODEL.FCPOSE.BASIS_MODULE.FOCAL_LOSS_ALPHA
            # self.focal_loss_gamma = cfg.MODEL.FCPOSE.BASIS_MODULE.FOCAL_LOSS_GAMMA

            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(conv_block(planes, planes, 3,1),
                                          conv_block(planes, planes, 3,1),)
            self.p3_logits = nn.Conv2d(planes, self.num_classes, kernel_size=1,
                                                    stride=1)
            self.upsampler = nn.Sequential(
                        ConvTranspose2d(planes+self.num_classes, planes, 8, stride=4, padding=6 // 2 - 1),
                        # get_norm(norm, planes),
                        nn.ReLU(),
                        # conv_block(planes, planes, 3,1),
                        )
            self.p1_logits = nn.Conv2d(planes, self.num_classes, kernel_size=3,
                                        stride=1, padding=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.p3_logits.bias, 0.0)
            torch.nn.init.normal_(self.p3_logits.weight, std=0.0001)
            torch.nn.init.constant_(self.p1_logits.bias, 0.0)
            torch.nn.init.normal_(self.p1_logits.weight, std=0.0001)
            # torch.nn.init.constant_(self.upsampler[0].bias, 0.0)
            # torch.nn.init.normal_(self.upsampler[0].weight, std=0.001)
            # torch.nn.init.constant_(self.upsampler[1].bias, 0.0)
            # torch.nn.init.constant_(self.upsampler[1].weight, 1.0)
            # torch.nn.init.normal_(self.upsampler[3][0].weight, std=0.0001)

    def forward(self, features, p1_heatmap_list=None, p3_heatmap_list=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}
        losses = {}
        # auxiliary thing semantic loss
        x = self.seg_head(x)
        p3_logits = self.p3_logits(x)
        outputs['basis_seg'] = p3_logits
        if self.training and self.loss_on:
            x = torch.cat([x, p3_logits], dim = 1)
            x = self.upsampler(x)
            p1_logits = self.p1_logits(x)
            p1_loss,p3_loss = compute_loss(p1_heatmap_list, p3_heatmap_list, p1_logits, p3_logits)
            losses['p1_loss'] = p1_loss * self.heatmap_loss_weight
            losses['p3_loss'] = p3_loss * self.heatmap_loss_weight
        return outputs, losses



