import math
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, IOULoss, NaiveGroupNorm, GCN
from .MEInst_outputs import MEInstOutputs
from .MaskEncoding import PCAMaskEncoding


__all__ = ["MEInst"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class MEInst(nn.Module):
    """
    Implement MEInst (https://arxiv.org/abs/2003.11712).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.cfg                  = cfg
        self.in_features          = cfg.MODEL.MEInst.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.MEInst.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.MEInst.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.MEInst.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.MEInst.CENTER_SAMPLE
        self.strides              = cfg.MODEL.MEInst.FPN_STRIDES
        self.radius               = cfg.MODEL.MEInst.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.MEInst.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.MEInst.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.MEInst.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.MEInst.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.MEInst.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.MEInst.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.MEInst.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.MEInst.THRESH_WITH_CTR
        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.MEInst.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.MEInst.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.MEInst_head = MEInstHead(cfg, [input_shape[f] for f in self.in_features])

        self.flag_parameters = cfg.MODEL.MEInst.FLAG_PARAMETERS
        self.mask_encoding = PCAMaskEncoding(cfg)

    def forward(self, images, features, gt_instances):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, bbox_towers, mask_regression = self.MEInst_head(features)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
            if not self.flag_parameters:
                # encoding parameters.
                components_path = self.cfg.MODEL.MEInst.PATH_COMPONENTS
                # update parameters.
                components_path = components_path.replace('agnosticTrue', 'agnostic' + str(self.cfg.MODEL.MEInst.AGNOSTIC))
                components_path = components_path.replace('whitenTrue', 'whiten' + str(self.cfg.MODEL.MEInst.WHITEN))
                components_path = components_path.replace('sigmoidTrue', 'sigmoid' + str(self.cfg.MODEL.MEInst.SIGMOID))
                components_path = components_path.replace('60', str(self.cfg.MODEL.MEInst.DIM_MASK))
                parameters = np.load(components_path)
                device = torch.device(self.cfg.MODEL.DEVICE)
                with torch.no_grad():
                    if self.cfg.MODEL.MEInst.AGNOSTIC:
                        components = nn.Parameter(torch.from_numpy(parameters['components_c'][0]).float().to(device),
                                                  requires_grad=False)
                        explained_variances = nn.Parameter(torch.from_numpy(parameters['explained_variance_c'][0])
                                                           .float().to(device), requires_grad=False)
                        means = nn.Parameter(torch.from_numpy(parameters['mean_c'][0]).float().to(device),
                                             requires_grad=False)
                        self.mask_encoding.components = components
                        self.mask_encoding.explained_variances = explained_variances
                        self.mask_encoding.means = means
                    else:
                        raise NotImplementedError
                self.flag_parameters = True
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = MEInstOutputs(
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            mask_regression,
            self.mask_encoding,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.MEInst_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances,
            cfg=self.cfg
        )

        if self.training:
            losses, _ = outputs.losses()
            return None, losses
        else:
            proposals = outputs.predict_proposals()
            return proposals, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    @staticmethod
    def compute_locations_per_level(h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class MEInstHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.MEInst.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.MEInst.FPN_STRIDES
        self.dim_mask = cfg.MODEL.MEInst.DIM_MASK
        self.use_gcn_in_mask = cfg.MODEL.MEInst.USE_GCN_IN_MASK
        self.gcn_kernel_size = cfg.MODEL.MEInst.GCN_KERNEL_SIZE

        head_configs = {"cls": (cfg.MODEL.MEInst.NUM_CLS_CONVS,
                                cfg.MODEL.MEInst.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.MEInst.NUM_BOX_CONVS,
                                 cfg.MODEL.MEInst.USE_DEFORMABLE),
                        "share": (cfg.MODEL.MEInst.NUM_SHARE_CONVS,
                                  cfg.MODEL.MEInst.USE_DEFORMABLE),
                        "mask": (cfg.MODEL.MEInst.NUM_MASK_CONVS,
                                 cfg.MODEL.MEInst.USE_DEFORMABLE)}

        self.type_deformable = cfg.MODEL.MEInst.TYPE_DEFORMABLE
        self.last_deformable = cfg.MODEL.MEInst.LAST_DEFORMABLE
        norm = None if cfg.MODEL.MEInst.NORM == "none" else cfg.MODEL.MEInst.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                # conv type.
                if use_deformable:
                    if self.last_deformable:
                        if i == num_convs - 1:
                            conv_func = DFConv2d
                            type_func = self.type_deformable
                        else:
                            conv_func = nn.Conv2d
                            type_func = "Conv2d"
                    else:
                        conv_func = DFConv2d
                        type_func = self.type_deformable
                else:
                    conv_func = nn.Conv2d
                    type_func = "Conv2d"
                # conv operation.
                if type_func == "DCNv1":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False,
                        with_modulated_dcn=False
                    ))
                elif type_func == "DCNv2":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=False,
                        with_modulated_dcn=True
                    ))
                elif type_func == "Conv2d":
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                else:
                    raise NotImplementedError
                # norm.
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                # activation.
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if self.use_gcn_in_mask:
            self.mask_pred = GCN(in_channels, self.dim_mask, k=self.gcn_kernel_size)
        else:
            self.mask_pred = nn.Conv2d(
                in_channels, self.dim_mask, kernel_size=3,
                stride=1, padding=1
            )

        if cfg.MODEL.MEInst.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness,
            self.mask_tower, self.mask_pred
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.MEInst.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        bbox_towers = []
        mask_reg = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved MEInst, instead of exp.
            bbox_reg.append(F.relu(reg))

            # Mask Encoding
            mask_tower = self.mask_tower(feature)
            mask_reg.append(self.mask_pred(mask_tower))

        return logits, bbox_reg, ctrness, bbox_towers, mask_reg
