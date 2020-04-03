# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch.nn as nn

from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone

from .lpf import *


__all__ = ['ResNetLPF', 'build_resnet_lpf_backbone']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride == 1):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
                                       conv3x3(planes, planes),)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups)  # stride moved
        self.bn2 = norm_layer(planes)
        if(stride == 1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
                                       conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetLPF(Backbone):

    def __init__(self, cfg, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, filter_size=1,
                 pool_only=True, return_idx=[0, 1, 2, 3]):
        super().__init__()
        self.return_idx = return_idx
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
                                           Downsample(filt_size=filter_size, stride=2, channels=planes[0])])
        else:
            self.maxpool = nn.Sequential(*[Downsample(filt_size=filter_size, stride=2, channels=planes[0]),
                                           nn.MaxPool2d(kernel_size=2, stride=1),
                                           Downsample(filt_size=filter_size, stride=2, channels=planes[0])])

        self.layer1 = self._make_layer(
            block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(
            block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer4 = self._make_layer(
            block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)
        if False:
            self._freeze_bn()

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                # stage 0 is the stem
                for p in self.conv1.parameters():
                    p.requires_grad = False
                for p in self.bn1.parameters():
                    p.requires_grad = False
            else:
                m = getattr(self, "layer" + str(stage_index))
                for p in m.parameters():
                    p.requires_grad = False

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [Downsample(filt_size=filter_size, stride=stride,
                                     channels=self.inplanes), ] if(stride != 1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                           norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups, norm_layer, filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups,
                                norm_layer=norm_layer, filter_size=filter_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        outs.append(self.layer1(x))  # 1/4
        outs.append(self.layer2(outs[-1]))  # 1/8
        outs.append(self.layer3(outs[-1]))  # 1/16
        outs.append(self.layer4(outs[-1]))  # 1/32
        return {"res{}".format(idx + 2): outs[idx] for idx in self.return_idx}


@BACKBONE_REGISTRY.register()
def build_resnet_lpf_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    depth = cfg.MODEL.RESNETS.DEPTH
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    out_stage_idx = [{"res2": 0, "res3": 1, "res4": 2, "res5": 3}[f] for f in out_features]
    out_feature_channels = {"res2": 256, "res3": 512,
                            "res4": 1024, "res5": 2048}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = ResNetLPF(cfg, Bottleneck, num_blocks_per_stage, norm_layer=NaiveSyncBatchNorm,
                      filter_size=3, pool_only=True, return_idx=out_stage_idx)
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model
