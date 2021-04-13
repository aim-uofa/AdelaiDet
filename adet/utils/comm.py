import torch
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.utils.comm import get_world_size


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious
