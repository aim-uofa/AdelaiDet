
"""
A working example to export the R-50 based FCOS model:
python onnx/test_onnxruntime.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --output /data/pretrained/onnx/fcos/FCOS_R_50_1x_bn_head.onnx
    --opts MODEL.WEIGHTS /data/pretrained/pytorch/fcos/FCOS_R_50_1x_bn_head.pth MODEL.FCOS.NORM "BN"

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
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
import numpy as np
import onnxruntime as rt

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
from collections import OrderedDict

from adet.config import get_cfg

class FCOS(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        return features

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

    # patch fcos_head
    # step 1. config
    fcos_head = model.proposal_generator.fcos_head
    norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
    head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                            cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                             cfg.MODEL.FCOS.USE_DEFORMABLE),
                    "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                              False)}

    # step 2. seperate module
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

    # step 3. override forward
    def fcos_head_forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = getattr(self, 'cls_tower{}'.format(l))(feature)
            bbox_tower = getattr(self, 'bbox_tower{}'.format(l))(feature)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))

        return logits, bbox_reg, ctrness, bbox_towers

    fcos_head.forward = types.MethodType(fcos_head_forward, fcos_head)

    proposal_generator = FCOS(cfg)
    onnx_model = torch.nn.Sequential(OrderedDict([
        ('backbone', model.backbone),
        ('proposal_generator', proposal_generator),
        ('heads', model.proposal_generator.fcos_head),
    ]))

    height, width = 800, 1088
    if args.width > 0:
        width = args.width
    if args.height > 0:
        height = args.height
    input_names = ["input_image"]
    dummy_input = torch.zeros((1, 3, height, width)).to(cfg.MODEL.DEVICE)
    output_names = []
    for item in ["logits", "bbox_reg", "centerness"]:
        for l in range(len(cfg.MODEL.FCOS.FPN_STRIDES)):
            fpn_name = "P{}".format(3 + l)
            output_names.extend([fpn_name + item])

    logger.info("Load onnx model from {}.".format(args.output))
    sess = rt.InferenceSession(args.output)
    
    for in_blob in sess.get_inputs():
        if in_blob.name not in input_names:
            print("Input blob name not match that in the mode")
        else:
            print("Input {}, shape {} and type {}".format(in_blob.name, in_blob.shape, in_blob.type))
    for out_blob in sess.get_outputs():
        if out_blob.name not in output_names:
            print("Output blob name not match that in the mode")
        else:
            print("Output {}, shape {} and type {}".format(out_blob.name, out_blob.shape, out_blob.type))

    with torch.no_grad():
        torch_output = onnx_model(dummy_input)
        logits, bbox_reg, ctrness, bbox_towers = torch_output
        lists = logits + bbox_reg + ctrness + bbox_towers

    onnx_output = sess.run(None, {input_names[0]: dummy_input.cpu().numpy()})
    for i, out in enumerate(onnx_output):
        try:
            np.testing.assert_allclose(lists[i].cpu().detach().numpy(), out, rtol=1e-03, atol=2e-04)
        except AssertionError as e:
            print("ouput {} mismatch {}".format(output_names[i], e))
            continue
        print("ouput {} match".format(output_names[i]))

if __name__ == "__main__":
    main()

