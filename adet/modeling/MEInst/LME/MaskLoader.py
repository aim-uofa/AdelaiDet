# coding:utf-8

import os
import json
import numpy as np

import torch.utils.data as data

from detectron2.structures import (
    Boxes,
    PolygonMasks,
    BoxMode
)


DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        }
}


class MaskLoader(data.Dataset):
    """
    Dataloader for Local Mask.

    Arguments:
        root (string): filepath to dataset folder.
        dataset (string): mask to use (eg. 'train', 'val').
        size (tuple): The size used for train/val (height, width).
        transform (callable, optional): transformation to perform on the input mask.

    """

    def __init__(self, root="datasets", dataset="coco_2017_train", size=28, transform=False):
        self.root = root
        self.dataset = dataset
        self.transform = transform

        if isinstance(size, int):
            self.size = size
        else:
            raise TypeError

        data_info = DATASETS[dataset]
        img_dir, ann_file = data_info['img_dir'], data_info['ann_file']
        img_dir = os.path.join(self.root, img_dir)  # actually we do not use it.
        ann_file = os.path.join(self.root, ann_file)

        with open(ann_file, 'r') as f:
            anns = json.load(f)
        anns = anns['annotations']
        coco = list()
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:
                coco.append(ann)
        self.coco = coco
        print("Removed {} images with no usable annotations. {} images left.".format(
              len(anns) - len(self.coco), len(self.coco)))

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, index):
        ann = self.coco[index]

        # bbox transform.
        bbox = np.array([ann["bbox"]])  # xmin, ymin, w, h
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)  # x1y1x2y2
        bbox = Boxes(bbox)

        # mask transform.
        mask = PolygonMasks([ann["segmentation"]])
        mask = mask.crop_and_resize(bbox.tensor, self.size).float()

        return mask
