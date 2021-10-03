import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from adet.modeling.fcos import FCOS
from .basis_module import basis_module
from .fcpose_head import fcpose_head_module
from .utils import compute_basis_stride, top_module, process_gt_instances



__all__ = ["FCPose"]



@PROPOSAL_GENERATOR_REGISTRY.register()
class FCPose(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.fcos = FCOS(cfg, input_shape)
        self.top_module = top_module(256, cfg.MODEL.FCPOSE.ATTN_LEN)

        self.basis_module = basis_module(cfg,input_shape)

        self.fcpose_head = fcpose_head_module(cfg)

        self.gt_stride = cfg.MODEL.FCPOSE.GT_HEATMAP_STRIDE
        self.device    = cfg.MODEL.DEVICE

    def forward(self, images, features, gt_instances=None):
        if gt_instances is not None:
            basis_gt_heatmap, head_gt_heatmap,p3_heatmap_list = process_gt_instances(gt_instances, self.gt_stride, self.device)
        else:
            basis_gt_heatmap, head_gt_heatmap,p3_heatmap_list = None, None, None

        proposals, proposal_losses = self.fcos(images, features, gt_instances, self.top_module)


        basis_out, basis_losses = self.basis_module(features, basis_gt_heatmap, p3_heatmap_list)
        del features, basis_gt_heatmap, p3_heatmap_list


        # base_stride = compute_basis_stride(images, basis_out)
        detector_results, detector_losses = self.fcpose_head(
            basis_out["bases"], proposals,
            head_gt_heatmap, gt_instances, basis_out["basis_seg"]
        )

        losses = {}
        if self.training:
            losses.update(proposal_losses)
            losses.update(basis_losses)
            losses.update(detector_losses)


        return detector_results, losses