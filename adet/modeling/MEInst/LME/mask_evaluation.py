# coding:utf-8

import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from MaskLoader import MaskLoader
from utils import (
    IOUMetric,
    transform,
    inverse_transform,
    direct_sigmoid,
    inverse_sigmoid
)


VALUE_MAX = 0.05
VALUE_MIN = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for PCA Mask Encoding.')
    parser.add_argument('--root', default='datasets', type=str)
    parser.add_argument('--dataset', default='coco_2017_train', type=str)
    parser.add_argument('--matrix', default='coco/components/coco_2017_train'
                        '_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz', type=str)
    # mask encoding params.
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_components', default=60, type=int)
    parser.add_argument('--class_agnostic', default=True, type=bool)
    parser.add_argument('--whiten', default=True, type=bool)
    parser.add_argument('--sigmoid', default=True, type=bool)
    parser.add_argument('--batch-size', default=1024, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # parse args.
    mask_size = args.mask_size
    n_components = args.n_components
    class_agnostic = args.class_agnostic
    whiten = args.whiten
    sigmoid = args.sigmoid

    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find("AdelaiDet") + len("AdelaiDet")]
    dataset_root = os.path.join(root_path, args.root)
    matrix_path = os.path.join(dataset_root, args.matrix)

    # load matrix.
    print("Loading matrix parameters: {}".format(matrix_path))
    parameters = np.load(matrix_path)
    components_c = parameters['components_c']
    mean_c = parameters['mean_c']
    ratio_c = parameters['ratio_c']
    explained_variance_c = parameters['explained_variance_c']
    if class_agnostic:
        components_c = np.squeeze(components_c)
        mean_c = np.squeeze(mean_c)
        explained_variance_c = np.squeeze(explained_variance_c)
        assert n_components == components_c.shape[0], \
            print("The n_components in component_ must equal to the supposed shape.")
    else:
        # TODO: We have not achieve the function in class-specific.
        raise NotImplementedError

    # build data loader.
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset, size=mask_size)
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    size_data = len(mask_loader)

    # evaluation.
    IoUevaluate = IOUMetric(2)
    print("Start Eva ...")
    for i, masks in enumerate(mask_loader):
        print("Eva [{} / {}]".format(i, size_data))
        # generate the reconstruction mask.
        masks = masks.view(masks.shape[0], -1).numpy()
        masks = masks.astype(np.float32)
        # pre-process.
        if sigmoid:
            value_random = VALUE_MAX * np.random.rand(masks.shape[0], masks.shape[1])
            value_random = np.maximum(value_random, VALUE_MIN)
            masks_random = np.where(masks > value_random, 1 - value_random, value_random)
            masks_random = inverse_sigmoid(masks_random)
        else:
            masks_random = masks
        # --> encode --> decode.
        mask_rc = transform(masks_random, components_=components_c, explained_variance_=explained_variance_c,
                            mean_=mean_c, whiten=whiten)
        mask_rc = inverse_transform(mask_rc, components_=components_c, explained_variance_=explained_variance_c,
                                    mean_=mean_c, whiten=whiten)
        # post-process.
        if sigmoid:
            mask_rc = direct_sigmoid(mask_rc)
        # eva.
        mask_rc = np.where(mask_rc >= 0.5, 1, 0)
        IoUevaluate.add_batch(mask_rc, masks)

    _, _, _, mean_iu, _ = IoUevaluate.evaluate()
    print("The mIoU for {}: {}".format(args.matrix, mean_iu))
