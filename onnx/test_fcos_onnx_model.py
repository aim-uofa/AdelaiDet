"""
"""

import onnx
import caffe2.python.onnx.backend as backend
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
    import sys
    print(sys.path)

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

class ONNX(torch.nn.Module):
    def __init__(self, onnx_model_path, cfg):
        super(ONNX, self).__init__()
        self.onnx_model = backend.prepare(
            onnx.load(onnx_model_path),
            device=cfg.MODEL.DEVICE.upper()
        )

    def forward(self, images):
        outputs = self.onnx_model.run(images.cpu().numpy())
        outputs = [torch.from_numpy(o).to(self.cfg.MODEL.DEVICE) for o in outputs]
        num_outputs = len(outputs) // 3
        logits = outputs[:num_outputs]
        bbox_reg = outputs[num_outputs:2 * num_outputs]
        centerness = outputs[2 * num_outputs:]
        return logits, bbox_reg, centerness

def main():
    parser = argparse.ArgumentParser(description="Test onnx models of FCOS")
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
    torch_output = onnx_model.forward(dummy_input)
    #logits, bbox_reg, ctrness, bbox_towers = torch_output


    onnx_model = ONNX(args.output, cfg)
    onnx_model.to(cfg.MODEL.DEVICE)
    onnx_result = onnx_model.forward(dummy_input)
if __name__ == "__main__":
    main()
