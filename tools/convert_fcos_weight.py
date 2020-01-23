import argparse
from collections import OrderedDict

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="FCOS Detectron2 Converter")
    parser.add_argument(
        "--model",
        default="weights/fcos_R_50_1x_official.pth",
        metavar="FILE",
        help="path to model weights",
    )
    parser.add_argument(
        "--output",
        default="weights/fcos_R_50_1x_converted.pth",
        metavar="FILE",
        help="path to model weights",
    )
    return parser


def rename_resnet_param_names(ckpt_state_dict):
    converted_state_dict = OrderedDict()
    for key in ckpt_state_dict.keys():
        value = ckpt_state_dict[key]

        key = key.replace("module.", "")
        key = key.replace("body", "bottom_up")

        # adding a . ahead to avoid renaming the fpn modules
        # this can happen after fpn renaming
        key = key.replace(".layer1", ".res2")
        key = key.replace(".layer2", ".res3")
        key = key.replace(".layer3", ".res4")
        key = key.replace(".layer4", ".res5")
        key = key.replace("downsample.0", "shortcut")
        key = key.replace("downsample.1", "shortcut.norm")
        key = key.replace("bn1", "conv1.norm")
        key = key.replace("bn2", "conv2.norm")
        key = key.replace("bn3", "conv3.norm")
        key = key.replace("fpn_inner2", "fpn_lateral3")
        key = key.replace("fpn_inner3", "fpn_lateral4")
        key = key.replace("fpn_inner4", "fpn_lateral5")
        key = key.replace("fpn_layer2", "fpn_output3")
        key = key.replace("fpn_layer3", "fpn_output4")
        key = key.replace("fpn_layer4", "fpn_output5")
        key = key.replace("top_blocks", "top_block")
        key = key.replace("fpn.", "")
        key = key.replace("rpn", "proposal_generator")
        key = key.replace("head", "fcos_head")

        converted_state_dict[key] = value
    return converted_state_dict


if __name__ == "__main__":
    args = get_parser().parse_args()
    ckpt = torch.load(args.model)
    model = rename_resnet_param_names(ckpt["model"])
    torch.save(model, args.output)
