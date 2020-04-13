"""
A working example to export the R-50 based FCOS model:
python onnx/export_model_to_onnx.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    MODEL.WEIGHTS weights/fcos_R_50_1x.pth

"""

import argparse
import os
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

try:
    from remove_python_path import remove_path
    remove_path()
except:
    pass
    #import sys
    #print(sys.path)

import torch
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

    # The onnx model can only be used with DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.NUM_WORKERS = 0

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

    proposal_generator = FCOS(cfg)
    onnx_model = torch.nn.Sequential(OrderedDict([
        ('backbone', model.backbone),
        ('proposal_generator', proposal_generator),
        ('heads', model.proposal_generator.fcos_head),
    ]))

    height, width = 512, 640
    input_names = ["input_image"]
    dummy_input = torch.zeros((1, 3, height, width)).to(cfg.MODEL.DEVICE)
    output_names = []
    for l in range(len(cfg.MODEL.FCOS.FPN_STRIDES)):
        fpn_name = "P{}/".format(3 + l)
        output_names.extend([
            fpn_name + "logits",
            fpn_name + "bbox_reg",
            fpn_name + "centerness"
        ])

    torch.onnx.export(
        onnx_model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=True
    )

    logger.info("Done. The onnx model is saved into {}.".format(args.output))


if __name__ == "__main__":
    main()
