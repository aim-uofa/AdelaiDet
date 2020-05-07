# coding:utf-8

import argparse
import os
import time
import json
import numpy as np

from sklearn.decomposition import IncrementalPCA
from pca_utils import inverse_sigmoid


VALUE_MAX = 0.05
VALUE_MIN = 0.01


def pca_compute(masks, n_components=60, class_agnostic=True, whiten=True, sigmoid=True, mask_size=28):
    components_c = []
    mean_c = []
    ratio_c = []
    explained_variance_c = []
    if class_agnostic:
        masks = np.concatenate([np.array(mask).astype(np.float32).reshape((-1, mask_size**2)) for mask in masks])
        if sigmoid:
            value_random = VALUE_MAX * np.random.rand(masks.shape[0], masks.shape[1])
            value_random = np.maximum(value_random, VALUE_MIN)
            masks = np.where(masks > value_random, 1-value_random, value_random)
            masks = inverse_sigmoid(masks)
        pca = IncrementalPCA(n_components=n_components, copy=False, whiten=whiten, batch_size=1024)
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


def parse_json(file_path):
    # load json and parse attributes.
    with open(file_path, 'r') as f:
        datasets = json.load(f)
    id_category = datasets['category_id']
    num_category = datasets['category_num']
    masks = datasets['segmentation_masks']
    return id_category, num_category, masks


if __name__ == '__main__':
    # ArgumentParser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/mnt/cephfs_new_wj/mlnlp/zhangrufeng/projects/'
                                                'adet/datasets/28x28/components/', type=str)
    parser.add_argument('--gt_set', default='/mnt/cephfs_new_wj/mlnlp/zhangrufeng/projects/'
                                            'adet/datasets/28x28/'
                                            'coco_2017_train/coco_2017_train_%s.json', type=str)
    parser.add_argument('--n_split', default=4, type=int)  # the number of split.
    # set it with a large number if you encounter OOM/ e.g. == n_split
    parser.add_argument('--pca_split', default=1, type=int)
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_components', default=60, type=int)
    parser.add_argument('--class_agnostic', default=True, type=bool)
    parser.add_argument('--whiten', default=True, type=bool)
    parser.add_argument('--sigmoid', default=True, type=bool)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # parse args.
    gt_set = args.gt_set
    n_split = args.n_split
    pca_split = args.pca_split
    mask_size = args.mask_size
    n_components = args.n_components
    class_agnostic = args.class_agnostic
    whiten = args.whiten
    sigmoid = args.sigmoid

    if pca_split > 1:
        len_split = n_split // pca_split
        splits = [len_split*_ for _ in range(pca_split)]

        components_c = []
        mean_c = []
        ratio_c = []
        explained_variance_c = []
        ratio = []

        masks = []
        for split in range(n_split):
            print("Start to Load the Segmentation Masks Split %d."%(split+1))
            gt_set_split = gt_set%str(split+1)
            tic = time.time()
            _, _, masks_split = parse_json(gt_set_split)
            toc = time.time() - tic
            if n_split == pca_split:
                masks = masks_split
            else:
                masks += masks_split
            print("Finish the load of split-%d in %2fs." % (split+1, toc))
            if (split + 1) in splits[1:]:
                print("Start to compute pca split...")
                tic = time.time()
                components_split, mean_split, ratio_split, explained_variance_split, ratio_s = \
                    pca_compute(masks, n_components=n_components, class_agnostic=class_agnostic,
                                whiten=whiten, sigmoid=sigmoid, mask_size=mask_size)
                toc = time.time() - tic
                print("Finish the pca computation in %2fs." % toc)
                masks = []
                components_c += components_split
                mean_c += mean_split
                ratio_c += ratio_split
                explained_variance_c += explained_variance_split
                ratio.append(ratio_s)

        print("Start to compute last pca split...")
        tic = time.time()
        components_split, mean_split, ratio_split, explained_variance_split, ratio_s = \
            pca_compute(masks, n_components=n_components, class_agnostic=class_agnostic,
                        whiten=whiten, sigmoid=sigmoid, mask_size=mask_size)
        toc = time.time() - tic
        print("Finish the last pca computation in %2fs." % toc)
        components_c += components_split
        mean_c += mean_split
        ratio_c += ratio_split
        explained_variance_c += explained_variance_split
        ratio.append(ratio_s)
    else:
        masks = []
        for split in range(n_split):
            print("Start to Load the Segmentation Masks Split %d." % (split + 1))
            gt_set_split = gt_set % str(split + 1)
            tic = time.time()
            _, _, masks_split = parse_json(gt_set_split)
            toc = time.time() - tic
            masks += masks_split
            print("Finish the load of split-%d in %2fs." % (split + 1, toc))

        print("Start to compute pca...")
        tic = time.time()
        components_c, mean_c, ratio_c, explained_variance_c, ratio = \
            pca_compute(masks, n_components=n_components, class_agnostic=class_agnostic,
                        whiten=whiten, sigmoid=sigmoid, mask_size=mask_size)
        toc = time.time() - tic
        print("Finish the pca computation in %2fs." % toc)

    components_c = np.concatenate(components_c).mean(0)[np.newaxis, :, :].astype(np.float32)
    mean_c = np.concatenate(mean_c).mean(0)[np.newaxis, :].astype(np.float32)
    ratio_c = np.concatenate(ratio_c).mean(0)[np.newaxis, :].astype(np.float32)
    explained_variance_c = np.concatenate(explained_variance_c).mean(0)[np.newaxis, :].astype(np.float32)

    print("The mean variance_ratio for all categories is %2f" % np.mean(ratio))

    # save the parameters
    output_path = os.path.join(output_dir, os.path.basename(gt_set).split('.')[0][:-3] + '_class_agnostic' +
                                str(class_agnostic) + '_whiten' + str(whiten) + '_sigmoid' + str(sigmoid) +
                                '_' + str(n_components) + '.npz')
    print("Save the pca results: " + output_path)
    np.savez(output_path,
             components_c=components_c,
             mean_c=mean_c,
             ratio_c=ratio_c,
             explained_variance_c=explained_variance_c)
