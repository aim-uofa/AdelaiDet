# coding:utf-8

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA

from MaskLoader import MaskLoader
from utils import inverse_sigmoid


VALUE_MAX = 0.05
VALUE_MIN = 0.01


def mask_encoding(masks, n_components=60, class_agnostic=True, whiten=True, sigmoid=True, batch_size=1024):
    components_c = []
    mean_c = []
    ratio_c = []
    explained_variance_c = []
    if class_agnostic:
        if sigmoid:
            value_random = VALUE_MAX * np.random.rand(masks.shape[0], masks.shape[1])
            value_random = np.maximum(value_random, VALUE_MIN)
            masks = np.where(masks > value_random, 1-value_random, value_random)
            masks = inverse_sigmoid(masks)
        pca = IncrementalPCA(n_components=n_components, copy=False, whiten=whiten, batch_size=batch_size)
        pca.fit(masks)
        components_c.append(pca.components_[np.newaxis, :, :])
        mean_c.append(pca.mean_[np.newaxis, :])
        ratio_c.append(pca.explained_variance_ratio_[np.newaxis, :])
        explained_variance_c.append(pca.explained_variance_[np.newaxis, :])
        ratio = pca.explained_variance_ratio_.sum()
    else:
        # TODO: We have not achieve the function in class-specific.
        raise NotImplemented

    return components_c, mean_c, ratio_c, explained_variance_c, ratio


def parse_args():
    parser = argparse.ArgumentParser(description='PCA Mask Encoding for local mask.')
    parser.add_argument('--root', default='datasets', type=str)
    parser.add_argument('--dataset', default='coco_2017_train', type=str)
    parser.add_argument('--output', default='coco/components', type=str)
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
    output_dir = os.path.join(dataset_root, args.output)
    os.makedirs(output_dir, exist_ok=True)

    # build data loader.
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset, size=mask_size)
    mask_loader = DataLoader(mask_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # loading masks.
    masks = list()
    print("Start Loading Masks.")
    tic = time.time()
    for mask in mask_loader:
        masks.append(mask.squeeze(1))
    toc = time.time() - tic
    print("Finish Loading Masks in {}s.".format(toc))
    masks = torch.cat(masks, 0)
    masks = masks.view(masks.shape[0], -1).numpy()
    masks = masks.astype(np.float32)

    # mask encoding.
    print("Start to mask encoding ...")
    print("It may take several times, please wait patiently ...")
    tic = time.time()
    components_c, mean_c, ratio_c, explained_variance_c, ratio = \
        mask_encoding(masks, n_components, class_agnostic, whiten, sigmoid, args.batch_size)
    toc = time.time() - tic
    print("Finish the mask encoding in {}s.".format(toc))

    components_c = np.concatenate(components_c).mean(0)[np.newaxis, :, :].astype(np.float32)
    mean_c = np.concatenate(mean_c).mean(0)[np.newaxis, :].astype(np.float32)
    ratio_c = np.concatenate(ratio_c).mean(0)[np.newaxis, :].astype(np.float32)
    explained_variance_c = np.concatenate(explained_variance_c).mean(0)[np.newaxis, :].astype(np.float32)
    print("The mean variance_ratio for all categories is {}".format(np.mean(ratio)))

    # save the parameters.
    output_path = os.path.join(output_dir, args.dataset + '_class_agnostic' + str(class_agnostic)
                               + '_whiten' + str(whiten) + '_sigmoid' + str(sigmoid) + '_' + str(n_components)
                               + '.npz')
    print("Save the local mask encoding matrix: " + output_path)
    np.savez(output_path,
             components_c=components_c,
             mean_c=mean_c,
             ratio_c=ratio_c,
             explained_variance_c=explained_variance_c)
