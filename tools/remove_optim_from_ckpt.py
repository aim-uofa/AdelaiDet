import argparse

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "--path",
        default="output/person/blendmask/R_50_1x/",
        help="path to model weights",
    )
    parser.add_argument(
        "--name",
        default="R_50_1x.pth",
        help="name of output file",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    ckpt = torch.load(args.path + 'model_final.pth')
    model = ckpt["model"]
    torch.save(model, args.path + args.name)
