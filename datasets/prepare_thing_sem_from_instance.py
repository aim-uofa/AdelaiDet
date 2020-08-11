# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import functools
import multiprocessing as mp
import numpy as np
import os
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta


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


def _process_instance_to_semantic(anns, output_semantic, img, categories):
    img_size = (img["height"], img["width"])
    output = np.zeros(img_size, dtype=np.uint8)
    for ann in anns:
        mask = annToMask(ann, img_size)
        output[mask == 1] = categories[ann["category_id"]] + 1
    # save as compressed npz
    np.savez_compressed(output_semantic, mask=output)
    # Image.fromarray(output).save(output_semantic)


def create_coco_semantic_from_instance(instance_json, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to contiguous ids starting from 1, and maps all unlabeled pixels to class 0

    Args:
        instance_json (str): path to the instance json file, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (dict): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    coco_detection = COCO(instance_json)

    def iter_annotations():
        for img_id in coco_detection.getImgIds():
            anns_ids = coco_detection.getAnnIds(img_id)
            anns = coco_detection.loadAnns(anns_ids)
            img = coco_detection.loadImgs(int(img_id))[0]
            file_name = os.path.splitext(img["file_name"])[0]
            output = os.path.join(sem_seg_root, file_name + '.npz')
            yield anns, output, img

    # single process
    # print("Start writing to {} ...".format(sem_seg_root))
    # start = time.time()
    # for anno, oup, img in iter_annotations():
    #     _process_instance_to_semantic(
    #         anno, oup, img, categories)
    # print("Finished. time: {:.2f}s".format(time.time() - start))
    # return

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(
            _process_instance_to_semantic,
            categories=categories),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="dataset to generate",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    dataset_dir = os.path.join(os.path.dirname(__file__), args.dataset_name)
    if args.dataset_name == "coco":
        thing_id_to_contiguous_id = _get_coco_instances_meta()["thing_dataset_id_to_contiguous_id"]
        split_name = 'train2017'
        annotation_name = "annotations/instances_{}.json"
    else:
        thing_id_to_contiguous_id = {1: 0}
        split_name = 'train'
        annotation_name = "annotations/{}_person.json"
    for s in ["train2017"]:
        create_coco_semantic_from_instance(
            os.path.join(dataset_dir, "annotations/instances_{}.json".format(s)),
            os.path.join(dataset_dir, "thing_{}".format(s)),
            thing_id_to_contiguous_id
        )
