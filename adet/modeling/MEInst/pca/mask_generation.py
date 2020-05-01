# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import os
import json

from utils import COCODataset, project_masks_on_boxes


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

ROOT = os.path.join(os.getcwd(), "datasets")


def main():
    parser = argparse.ArgumentParser(description="Mask generation.")
    parser.add_argument("--dataset_name", type=str, default='coco_2017_train',
                        help="the name of dataset.")
    parser.add_argument("--output_dir", type=str, default='./',
                        help="the name of output dir.")
    parser.add_argument('--num_classes', type=int, default=81,
                        help="the number of classes (including background).")
    parser.add_argument('--discretization_size', type=int, default=28,
                        help="the uniform size for binary-class masks.")
    parser.add_argument('--split', type=int, default=4,
                        help="the number of dataset split.")

    args = parser.parse_args()

    # hyper parameter.
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    num_classes = args.num_classes
    discretization_size = args.discretization_size
    split = args.split

    # input path.
    dataset = DATASETS[dataset_name]
    img_dir = dataset["img_dir"]
    ann_file = dataset["ann_file"]
    img_dir = os.path.join(ROOT, img_dir)
    ann_file = os.path.join(ROOT, ann_file)

    # output path.
    output_dir = os.path.join(ROOT, output_dir, str(discretization_size)
                              + "x" + str(discretization_size), dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # process.
    dataset = COCODataset(ann_file=ann_file, root=img_dir, remove_images_without_annotations=True)
    len_dataset = len(dataset)
    len_split = len_dataset // split
    flags = [len_split * _ for _ in range(split)]
    flag = 0
    masks = [[] for _ in range(1, num_classes)]

    # iteration
    for _, batch in enumerate(dataset):
        print("Processing [%d/%d]" % (_, len_dataset))
        idx, target = batch
        label = target.get_field("labels").numpy()
        segmentation_poly = target.get_field("masks")
        segmentation_mask = project_masks_on_boxes(segmentation_poly, proposals=target, discretization_size=discretization_size)
        segmentation_mask = segmentation_mask.view(-1, discretization_size**2).float().numpy()
        for l, seg in zip(label, segmentation_mask):
            masks[l-1].append(seg.tolist())
        if _ in flags[1:]:
            flag += 1
            mask_json = dict()
            mask_json['category_id'] = [_ for _ in range(1, num_classes)]
            mask_json['category_num'] = [len(mask) for mask in masks]
            mask_json['segmentation_masks'] = masks
            print("Save file_%d" % flag)
            with open(os.path.join(output_dir, dataset_name + '_%s' % flag + '.json'), 'w') as f:
                json.dump(mask_json, f)
            masks = [[] for _ in range(1, num_classes)]
    # store the last split.
    flag += 1
    mask_json = dict()
    mask_json['category_id'] = [_ for _ in range(1, num_classes)]
    mask_json['category_num'] = [len(mask) for mask in masks]
    mask_json['segmentation_masks'] = masks
    print("Save file_%d" % flag)
    with open(os.path.join(output_dir, dataset_name + '_%s' % flag + '.json'), 'w') as f:
        json.dump(mask_json, f)
    print("Finish processing.")


if __name__ == "__main__":
    main()
