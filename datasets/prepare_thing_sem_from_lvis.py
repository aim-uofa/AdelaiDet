# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import functools
import multiprocessing as mp
import numpy as np
import os
from lvis import LVIS
from pycocotools import mask as maskUtils


def annToRLE(ann, img_size):
    h, w = img_size
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annToMask(ann, img_size):
    rle = annToRLE(ann, img_size)
    m = maskUtils.decode(rle)
    return m


def _process_instance_to_semantic(anns, output_semantic, img):
    img_size = (img["height"], img["width"])
    output = np.zeros(img_size, dtype=np.uint8)
    for ann in anns:
        mask = annToMask(ann, img_size)
        output[mask == 1] = ann["category_id"] // 5
    # save as compressed npz
    np.savez_compressed(output_semantic, mask=output)
    # Image.fromarray(output).save(output_semantic)


def create_lvis_semantic_from_instance(instance_json, sem_seg_root):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to contiguous ids starting from 1, and maps all unlabeled pixels to class 0

    Args:
        instance_json (str): path to the instance json file, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    lvis_detection = LVIS(instance_json)

    def iter_annotations():
        for img_id in lvis_detection.get_img_ids():
            anns_ids = lvis_detection.get_ann_ids([img_id])
            anns = lvis_detection.load_anns(anns_ids)
            img = lvis_detection.load_imgs([img_id])[0]
            file_name = os.path.splitext(img["file_name"])[0]
            output = os.path.join(sem_seg_root, file_name + '.npz')
            yield anns, output, img

    # # single process
    # print("Start writing to {} ...".format(sem_seg_root))
    # start = time.time()
    # for anno, oup, img in iter_annotations():
    #     _process_instance_to_semantic(
    #         anno, oup, img)
    # print("Finished. time: {:.2f}s".format(time.time() - start))
    # return

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(
            _process_instance_to_semantic),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "lvis")
    for s in ["train"]:
        create_lvis_semantic_from_instance(
            os.path.join(dataset_dir, "lvis_v0.5_{}.json".format(s)),
            os.path.join(dataset_dir, "thing_{}".format(s)),
        )
