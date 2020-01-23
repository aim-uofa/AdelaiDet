import torch
from detectron2.engine import default_argument_parser, default_setup

from adet.config import get_cfg
from adet.utils.measures import measure_model

from train_net import Trainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    model.eval().cuda()
    input_size = (3, 512, 512)
    image = torch.zeros(*input_size)
    batched_input = {"image": image}
    ops, params = measure_model(model, [batched_input])
    print('ops: {:.2f}G\tparams: {:.2f}M'.format(ops / 2**30, params / 2**20))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
