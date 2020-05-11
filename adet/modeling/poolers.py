import sys
import torch
from torch import nn
from detectron2.layers import cat

from detectron2.modeling.poolers import (
    ROIPooler, convert_boxes_to_pooler_format, assign_boxes_to_levels
)

from adet.layers import BezierAlign
from adet.structures import Beziers

__all__ = ["TopPooler"]


def _box_max_size(boxes):
    box = boxes.tensor
    max_size = torch.max(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    return max_size


def _bezier_height(beziers):
    beziers = beziers.tensor
    # compute the distance between the first and last control point
    p1 = beziers[:, :2]
    p2 = beziers[:, 14:]
    height = ((p1 - p2) ** 2).sum(dim=1).sqrt()
    return height

    
def assign_boxes_to_levels_by_metric(
        box_lists, min_level, max_level, canonical_box_size,
        canonical_level, metric_fn=_box_max_size):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[detectron2.structures.Boxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (shorter side).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = cat([metric_fn(boxes) for boxes in box_lists])
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def assign_boxes_to_levels_max(
        box_lists, min_level, max_level, canonical_box_size,
        canonical_level):
    return assign_boxes_to_levels_by_metric(
        box_lists, min_level, max_level, canonical_box_size,
        canonical_level, metric_fn=_box_max_size
    )


def assign_boxes_to_levels_bezier(
        box_lists, min_level, max_level, canonical_box_size,
        canonical_level):
    return assign_boxes_to_levels_by_metric(
        box_lists, min_level, max_level, canonical_box_size,
        canonical_level, metric_fn=_bezier_height
    )


class TopPooler(ROIPooler):
    """
    ROIPooler with option to assign level by max length. Used by top modules.
    """
    def __init__(self,
                 output_size,
                 scales,
                 sampling_ratio,
                 pooler_type,
                 canonical_box_size=224,
                 canonical_level=4,
                 assign_crit="area",):
        # to reuse the parent initialization, handle unsupported pooler types
        parent_pooler_type = "ROIAlign" if pooler_type == "BezierAlign" else pooler_type
        super().__init__(output_size, scales, sampling_ratio, parent_pooler_type,
                         canonical_box_size=canonical_box_size,
                         canonical_level=canonical_level)
        if parent_pooler_type != pooler_type:
            # reinit the level_poolers here
            self.level_poolers = nn.ModuleList(
                BezierAlign(
                    output_size, spatial_scale=scale,
                    sampling_ratio=sampling_ratio) for scale in scales
            )
        self.assign_crit = assign_crit

    def forward(self, x, box_lists):
        """
        see 
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        if isinstance(box_lists[0], torch.Tensor):
            # TODO: use Beziers for data_mapper
            box_lists = [Beziers(x) for x in box_lists]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        if self.assign_crit == "max":
            assign_method = assign_boxes_to_levels_max
        elif self.assign_crit == "bezier":
            assign_method = assign_boxes_to_levels_bezier
        else:
            assign_method = assign_boxes_to_levels

        level_assignments = assign_method(
            box_lists, self.min_level, self.max_level,
            self.canonical_box_size, self.canonical_level)

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size[0], output_size[1]), dtype=dtype, device=device
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output
