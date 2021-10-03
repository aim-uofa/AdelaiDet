import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno

import math

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_beziers_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)

    return instance


def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""



class HeatmapGenerator():
    def __init__(self, num_joints, sigma, head_sigma):
        self.num_joints = num_joints
        self.sigma = sigma
        self.head_sigma = head_sigma

        self.p3_sigma = sigma / 2

        size = 2*np.round(3 * sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        size = 2*np.round(3 * self.p3_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.p3_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.p3_sigma ** 2))

        size = 2*np.round(3 * head_sigma) + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = (size - 1) /2, (size - 1) /2
        self.head_g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * head_sigma ** 2))

    def __call__(self, gt_instance, gt_heatmap_stride):
        heatmap_size = gt_instance.image_size
        heatmap_size = [math.ceil(heatmap_size[0]/ 32)*(32/gt_heatmap_stride),
                    math.ceil(heatmap_size[1]/ 32)*(32/gt_heatmap_stride)]

        h,w = heatmap_size
        h,w = int(h),int(w) 
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / gt_heatmap_stride
        sigma = self.sigma
        head_sigma = self.head_sigma
        p3_sigma = self.p3_sigma

        output_list = []
        head_output_list = []
        for p in joints:
            hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            head_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])

                    ul = int(np.round(x - 3 * head_sigma - 1)), int(np.round(y - 3 * head_sigma - 1))
                    br = int(np.round(x + 3 * head_sigma + 2)), int(np.round(y + 3 * head_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    head_hms[idx, aa:bb, cc:dd] = np.maximum(
                        head_hms[idx, aa:bb, cc:dd], self.head_g[a:b, c:d])
                    
            hms = torch.from_numpy(hms)
            head_hms = torch.from_numpy(head_hms)
            output_list.append(hms)
            head_output_list.append(head_hms)

        h,w = h//4, w//4
        p3_output_list = []
        joints = gt_instance.gt_keypoints.tensor.numpy().copy()
        joints[:,:,[0,1]] = joints[:,:,[0,1]] / 8
        for p in joints:
            p3_hms = np.zeros((self.num_joints, h, w),dtype=np.float32)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= w or y >= h:
                        continue

                    ul = int(np.round(x - 3 * p3_sigma - 1)), int(np.round(y - 3 * p3_sigma - 1))
                    br = int(np.round(x + 3 * p3_sigma + 2)), int(np.round(y + 3 * p3_sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], w)
                    aa, bb = max(0, ul[1]), min(br[1], h)
                    p3_hms[idx, aa:bb, cc:dd] = np.maximum(
                        p3_hms[idx, aa:bb, cc:dd], self.p3_g[a:b, c:d])
                    
            p3_hms = torch.from_numpy(p3_hms)
            p3_output_list.append(p3_hms)
        output_list = torch.stack(output_list,dim=0)
        p3_output_list = torch.stack(p3_output_list,dim=0)
        head_output_list = torch.stack(head_output_list,dim=0)
        gt_instance.keypoint_heatmap = output_list
        gt_instance.head_heatmap = head_output_list
        gt_instance.p3_output_list = p3_output_list
        return gt_instance