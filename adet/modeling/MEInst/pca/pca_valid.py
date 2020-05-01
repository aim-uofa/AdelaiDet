# coding:utf-8

import argparse
import os
import time
import numpy as np

from .pca_labels import parse_json
from .pca_utils import transform, inverse_transform, IOUMetric, direct_sigmoid, inverse_sigmoid

VALUE_MAX = 0.05
VALUE_MIN = 0.01


def pca_valid(masks, components_c,
              explained_variance_c, mean_c=None,
              n_components=60, class_agnostic=True, whiten=True, sigmoid=True, mask_size=28):

    mIoU = []
    if class_agnostic:
        masks = np.concatenate([np.array(mask).astype(np.float32).reshape((-1, mask_size**2)) for mask in masks])
        components_c = np.squeeze(components_c)
        mean_c = np.squeeze(mean_c)
        explained_variance_c = np.squeeze(explained_variance_c)
        assert n_components == components_c.shape[0], print(
            "The n_components in component_ must equal to the supposed shape.")
        # generate the reconstruction mask.
        if sigmoid:
            value_random = VALUE_MAX * np.random.rand(masks.shape[0], masks.shape[1])
            value_random = np.maximum(value_random, VALUE_MIN)
            masks_random = np.where(masks > value_random, 1 - value_random, value_random)
            masks_random = inverse_sigmoid(masks_random)
        else:
            masks_random = masks
        mask_rc = transform(masks_random, components_=components_c, explained_variance_=explained_variance_c, mean_=mean_c, whiten=whiten)
        mask_rc = inverse_transform(mask_rc, components_=components_c, explained_variance_=explained_variance_c, mean_=mean_c, whiten=whiten)
        if sigmoid:
            mask_rc = direct_sigmoid(mask_rc)
        mask_rc = np.where(mask_rc >= 0.5, 1, 0)
        IoUevaluate = IOUMetric(2)
        IoUevaluate.add_batch(mask_rc, masks)
        _, _, _, mean_iu, _ = IoUevaluate.evaluate()
        mIoU.append(mean_iu)
    else:
        # TODO: We have not achieve the function in class-specific.
        raise NotImplementedError
    return np.mean(mIoU)


if __name__ == '__main__':
    # ArgumentParser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_set', default='/mnt/cephfs_new_wj/mlnlp/zhangrufeng/projects/'
                                            'adet/datasets/28x28/coco_2017_train/coco_2017_train_%s.json', type=str)
    parser.add_argument('--output_dir', default='/mnt/cephfs_new_wj/mlnlp/zhangrufeng/projects/'
                                                'adet/datasets/28x28/components/', type=str)
    parser.add_argument('--n_split', default=4, type=int)
    parser.add_argument('--mask_size', default=28, type=int)
    parser.add_argument('--n_components', default=60, type=int)
    parser.add_argument('--class_agnostic', default=True, type=bool)
    parser.add_argument('--whiten', default=True, type=bool)
    parser.add_argument('--sigmoid', default=True, type=bool)
    parser.add_argument('--on_val', default=True, type=bool)

    args = parser.parse_args()
    gt_set = args.gt_set
    output_dir = args.output_dir
    n_split = args.n_split
    mask_size = args.mask_size
    n_components = args.n_components
    class_agnostic = args.class_agnostic
    whiten = args.whiten
    sigmoid = args.sigmoid
    on_val = args.on_val

    # load the parameters
    output_path = os.path.join(output_dir, os.path.basename(gt_set).split('.')[0][:-3] + '_class_agnostic' +
                               str(class_agnostic) + '_whiten' + str(whiten) + '_sigmoid' + str(sigmoid) +
                               '_' + str(n_components) + '.npz')
    print("Load the pca parameters: " + output_path)
    tic = time.time()
    parameters = np.load(output_path)
    components_c = parameters['components_c']
    mean_c = parameters['mean_c']
    ratio_c = parameters['ratio_c']
    explained_variance_c = parameters['explained_variance_c']
    toc = time.time() - tic
    print("Finish the load in %2fs."%toc)

    if on_val:
        gt_set = gt_set.replace('train', 'val')
    else:
        pass

    mIoU = []
    for split in range(n_split):
        print("Start to Load the Segmentation Masks Split %d." % (split + 1))
        gt_set_split = gt_set % str(split + 1)
        tic = time.time()
        _, _, masks = parse_json(gt_set_split)
        toc = time.time() - tic
        print("Finish the load of split-%d in %2fs." % (split + 1, toc))

        print("Start to valid pca of split-%d..."% (split + 1))
        mIoU_split = pca_valid(masks=masks, components_c=components_c,
                               explained_variance_c=explained_variance_c, mean_c=mean_c, n_components=n_components,
                               class_agnostic=class_agnostic, whiten=whiten, sigmoid=sigmoid, mask_size=mask_size)
        mIoU.append(mIoU_split)
        print("Finish the valid pca of split-%d"% (split + 1))

    print("The mIoU for %s is %f"%(output_path, np.mean(mIoU)))
