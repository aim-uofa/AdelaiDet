import argparse
from collections import OrderedDict

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="FCOS Detectron2 Converter")
    parser.add_argument(
        "--model",
        default="weights/blendmask/person/R_50_1x.pth",
        metavar="FILE",
        help="path to model weights",
    )
    parser.add_argument(
        "--output",
        default="weights/blendmask/person/R_50_1x.pth",
        metavar="FILE",
        help="path to model weights",
    )
    return parser


def rename_resnet_param_names(ckpt_state_dict):
    converted_state_dict = OrderedDict()
    for key in ckpt_state_dict.keys():
        value = ckpt_state_dict[key]
        key = key.replace("centerness", "ctrness")

        converted_state_dict[key] = value
    return converted_state_dict


if __name__ == "__main__":
    args = get_parser().parse_args()
    ckpt = torch.load(args.model)
    if "model" in ckpt:
        model = rename_resnet_param_names(ckpt["model"])
    else:
        model = rename_resnet_param_names(ckpt)
    torch.save(model, args.output)
