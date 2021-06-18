"""
A working example to export the R-50 based FCOS model:
python onnx/export_model_to_onnx.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --output /data/pretrained/onnx/fcos/FCOS_R_50_1x_bn_head.onnx
    --opts MODEL.WEIGHTS /data/pretrained/pytorch/fcos/FCOS_R_50_1x_bn_head.pth MODEL.FCOS.NORM "BN"

# about the upsample/interpolate
https://github.com/pytorch/pytorch/issues/10446
https://github.com/pytorch/pytorch/issues/18113
"""

import argparse
import os
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import types
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

# multiple versions of Adet/FCOS are installed, remove the conflict ones from the path
try:
    from remove_python_path import remove_path
    remove_path()
except:
    import sys
    print(sys.path)

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import ProposalNetwork

from adet.config import get_cfg
from adet.modeling import FCOS, BlendMask, BAText, MEInst, condinst, SOLOv2
from adet.modeling.condinst.mask_branch import MaskBranch

def patch_condinst(cfg, model, output_names):
    def forward(self, tensor):
        images = None
        gt_instances = None
        mask_feats = None
        proposals = None

        features = self.backbone(tensor)
        #return features
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.controller)
        #return proposals
        mask_feats, sem_losses = self.mask_branch(features, gt_instances)
        #return mask_feats
        return mask_feats, proposals

    model.forward = types.MethodType(forward, model)

    #output tensor naming [optional]

def patch_blendmask(cfg, model, output_names):
    def forward(self, tensor):
        images = None
        gt_instances = None
        basis_sem = None

        features = self.backbone(tensor)
        basis_out, basis_losses = self.basis_module(features, basis_sem)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.top_layer)
        return basis_out["bases"], proposals

    model.forward = types.MethodType(forward, model)

    #output tensor naming [optional]
    output_names.extend(["bases"])
    for item in ["logits", "bbox_reg", "centerness", "top_feats"]:
        for l in range(len(cfg.MODEL.FCOS.FPN_STRIDES)):
            fpn_name = "P{}".format(3 + l)
            output_names.extend([fpn_name + item])

def patch_ProposalNetwork(cfg, model, output_names):
    def forward(self, tensor):
        images = None
        gt_instances = None

        features = self.backbone(tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        return proposals

    model.forward = types.MethodType(forward, model)

    #output tensor naming [optional]
    for item in ["logits", "bbox_reg", "centerness"]:
        for l in range(len(cfg.MODEL.FCOS.FPN_STRIDES)):
            fpn_name = "P{}".format(3 + l)
            output_names.extend([fpn_name + item])

def patch_fcos(cfg, proposal_generator):
    def proposal_generator_forward(self, images, features, gt_instances=None, top_module=None):
        features = [features[f] for f in self.in_features]
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(features, top_module, self.yield_proposal)
        return (logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers), None

    proposal_generator.forward = types.MethodType(proposal_generator_forward, proposal_generator)

def patch_fcos_head(cfg, fcos_head):
    # step 1. config
    norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
    head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                            cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                             cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                              False)}

    # step 2. separate module
    for l in range(fcos_head.num_levels):
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 0]))
                if norm in ["GN", "NaiveGN"]:
                    tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 1]))
                elif norm in ["BN", "SyncBN"]:
                    tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 1][l]))
                tower.append(deepcopy(getattr(fcos_head, '{}_tower'.format(head))[i*3 + 2]))
            fcos_head.add_module('{}_tower{}'.format(head, l), torch.nn.Sequential(*tower))

    # step 3. override fcos_head forward
    def fcos_head_forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = getattr(self, 'cls_tower{}'.format(l))(feature)
            bbox_tower = getattr(self, 'bbox_tower{}'.format(l))(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))

        return logits, bbox_reg, ctrness, top_feats, bbox_towers

    fcos_head.forward = types.MethodType(fcos_head_forward, fcos_head)

def upsample(tensor, factor): # aligned_bilinear in adet/utils/comm.py is not onnx-friendly
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    oh = factor * h
    ow = factor * w
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='nearest',
    )
    return tensor

def patch_mask_branch(cfg, mask_branch):
    def mask_branch_forward(self, features, gt_instances=None):
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
                x_p = upsample(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        return mask_feats, losses

    mask_branch.forward = types.MethodType(mask_branch_forward, mask_branch)

def main():
    parser = argparse.ArgumentParser(description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="configs/FCOS-Detection/R_50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--width', default=0, type=int)
    parser.add_argument('--height', default=0, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument(
        "--output",
        default="output/fcos.onnx",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    cfg = get_cfg()
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # norm for ONNX: change FrozenBN back to BN
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.BASIS_MODULE.NORM = "BN"

    # turn on the following configuration according to your own convenience
    #cfg.MODEL.FCOS.NORM = "BN"
    #cfg.MODEL.FCOS.NORM = "NaiveGN"

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger(output=output_dir)
    logger.info(cfg)

    model = build_model(cfg)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    logger.info("Model:\n{}".format(model))

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info("load Model:\n{}".format(cfg.MODEL.WEIGHTS))

    height, width = 800, 1088
    if args.width > 0:
        width = args.width
    if args.height > 0:
        height = args.height
    input_names = ["input_image"]
    dummy_input = torch.zeros((1, 3, height, width)).to(cfg.MODEL.DEVICE)
    output_names = []
    if isinstance(model, condinst.CondInst):
        patch_condinst(cfg, model, output_names)

    if isinstance(model, BlendMask):
        patch_blendmask(cfg, model, output_names)

    if isinstance(model, ProposalNetwork):
        patch_ProposalNetwork(cfg, model, output_names)

    if hasattr(model, 'proposal_generator'):
        if isinstance(model.proposal_generator, FCOS):
            patch_fcos(cfg, model.proposal_generator)
            patch_fcos_head(cfg, model.proposal_generator.fcos_head)

    if hasattr(model, 'mask_branch'):
        if isinstance(model.mask_branch, MaskBranch):
            patch_mask_branch(cfg, model.mask_branch) # replace aligned_bilinear with nearest upsample

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=True,
    )

    logger.info("Done. The onnx model is saved into {}.".format(args.output))


if __name__ == "__main__":
    main()
