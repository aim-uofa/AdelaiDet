import torch.distributed as dist
from detectron2.utils.comm import get_world_size
from torch.nn import functional as F
from torch import nn
import torch
from detectron2.structures import ImageList
from adet.utils.comm import reduce_sum
from fvcore.nn import sigmoid_focal_loss_jit


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

def compute_basis_stride(images, basis_out):
    im_h, im_w = images.tensor.size()[-2:]
    assert len(basis_out["bases"]) == 1
    base_h, base_w = basis_out["bases"][0].size()[2:]
    base_stride_h, base_stride_w = im_h // base_h, im_w // base_w
    assert base_stride_h == base_stride_w
    base_stride = base_stride_w
    return base_stride

class folder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feature_map):
        N,_,H,W = feature_map.size()
        feature_map = F.unfold(feature_map,kernel_size=3,padding=1)
        feature_map = feature_map.view(N,-1,H,W)
        return feature_map

def top_module(in_channels, attn_len):
    return folder()

def process_gt_instances(gt_instances, gt_stride, device):
    basis_heatmap_list = []
    head_heatmap_list = []
    p3_heatmap_list = []
    for instances in gt_instances:
        one_frame_instances = instances.keypoint_heatmap.to(device = device, dtype = torch.float)
        one_basis_heatmap = one_frame_instances.max(dim = 0)[0]#.clamp(0,1)
        basis_heatmap_list.append(one_basis_heatmap)

        p3_output_list = instances.p3_output_list.to(device = device, dtype = torch.float)
        p3_output_list = p3_output_list.max(dim = 0)[0]#.clamp(0,1)
        p3_heatmap_list.append(p3_output_list)

        one_frame_instances = instances.head_heatmap.to(device = device, dtype = torch.float)
        for index_instence in range(len(instances)):
            head_heatmap_list.append(one_frame_instances[index_instence])
    basis_heatmap_list = ImageList.from_tensors(basis_heatmap_list)
    p3_heatmap_list = ImageList.from_tensors(p3_heatmap_list)
    head_heatmap_list  = ImageList.from_tensors(head_heatmap_list)
    return basis_heatmap_list.tensor, head_heatmap_list.tensor.bool(), p3_heatmap_list.tensor,


def compute_loss(p1_heatmap_list, p3_heatmap_list, p1_logits, p3_logits):
    # gt_bitmasks = gt_bitmasks.float()
    # mask_logits = mask_logits.sigmoid() 
    num_gpus = get_world_size()

    num_dice = (p1_heatmap_list**2).sum()
    num_dice = reduce_sum(p1_logits.new_tensor([num_dice])).item()
    num_dice = max(num_dice / num_gpus, 1.0)

    p1_loss = F.mse_loss(p1_heatmap_list, p1_logits, reduction='sum') / num_dice

    num_dice = (p3_heatmap_list**2).sum()
    num_dice = reduce_sum(p3_logits.new_tensor([num_dice])).item()
    num_dice = max(num_dice / num_gpus, 1.0)

    p3_loss = F.mse_loss(p3_heatmap_list, p3_logits, reduction='sum') / num_dice

    # loss = (p1_loss + p3_loss) / 2

    return p1_loss, p3_loss

def compute_loss_softmax(gt_bitmasks, mask_logits, num_loss, num_instances, direction, direction_mask_logits, gt_keypoint, max_ranges, distance_norm):
    assert not torch.isnan(mask_logits).any()
    assert not torch.isnan(direction).any()
    assert not torch.isnan(direction_mask_logits).any()
    # direction_mask_logits = direction_mask_logits.detach()
    N,K,H,W = gt_bitmasks.size()
    # gt_bitmasks = gt_bitmasks.float()
    num_gpus = get_world_size()
    assert not (num_loss == 0).any()
    loss_weight = 1/num_loss #TODO num_loss can be 0
    sum_loss_weight = loss_weight.sum()
    assert sum_loss_weight!=0
    loss_weight = loss_weight[:,None].repeat(1,17).flatten()
    
    gt_bitmasks = gt_bitmasks.reshape(N*K,H*W)
    mask_logits = mask_logits.reshape(N*K,H*W)
    gt_bitmasks_visible_mask = gt_bitmasks.sum(dim=1).bool()
    # assert gt_bitmasks_visible_mask.sum()!=0 #TODO AssertionError
    if gt_bitmasks_visible_mask.sum()!=0:
        loss_weight = loss_weight[gt_bitmasks_visible_mask]
        mask_logits = mask_logits[gt_bitmasks_visible_mask]
        gt_bitmasks = gt_bitmasks[gt_bitmasks_visible_mask]
        mask_logits = F.log_softmax(mask_logits,dim=1)

        total_instances = reduce_sum(mask_logits.new_tensor([num_instances])).item()
        gpu_balence_factor = num_instances/total_instances

        loss = (- mask_logits[gt_bitmasks])
        loss = (loss*loss_weight).sum()/17
        loss = (loss/sum_loss_weight)*gpu_balence_factor

        max_ranges = max_ranges[:,None].repeat(1,17).flatten()[gt_bitmasks_visible_mask]
        gt_keypoint = gt_keypoint[:,:,[0,1]]

        N,H,W,K,_ = direction_mask_logits.size()
        direction = direction - gt_keypoint[:,None,None,:,:]
        direction = direction.permute(0,3,1,2,4).reshape(N*17,H,W,2)
        direction = direction[gt_bitmasks_visible_mask]
        direction = (direction[:,:,:,0] ** 2 + direction[:,:,:,1] ** 2).sqrt()[:,:,:,None]
        assert (max_ranges != 0).all()
        direction = direction / max_ranges[:,None,None,None]
        direction = direction * distance_norm
        direction = (direction.sigmoid()-0.5) * 2
        direction_mask_logits = direction_mask_logits.permute(0,3,1,2,4).reshape(N*17,H,W,1)
        direction_mask_logits = direction_mask_logits[gt_bitmasks_visible_mask]
        direction = direction * direction_mask_logits
        direction = direction.flatten(start_dim=1).sum(dim=1)
        direction = direction * loss_weight
        assert distance_norm != 0
        direction_loss = (direction/sum_loss_weight * gpu_balence_factor) / distance_norm
        direction_loss = direction_loss.sum()
        assert not torch.isnan(direction_loss).any()
        assert not torch.isnan(loss).any()
        return loss, direction_loss
    else:
        print('gt_bitmasks_visible_mask.sum()==0')
        total_instances = reduce_sum(mask_logits.new_tensor([num_instances])).item()
        loss = mask_logits.sum() + direction.sum() + direction_mask_logits.sum()
        loss = loss*0.0
        return loss, loss


# def compute_loss(gt_bitmasks, mask_logits):
#     # assert torch.isfinite(gt_bitmasks).all() and torch.isfinite(mask_logits).all()
#     gt_bitmasks = gt_bitmasks.float()
#     num_gpus = get_world_size()

#     num_dice = gt_bitmasks.sum()
#     num_dice = reduce_sum(mask_logits.new_tensor([num_dice])).item()
#     num_dice = max(num_dice / num_gpus, 1.0)


#     loss = F.mse_loss(mask_logits, gt_bitmasks, reduction='sum') / num_dice
    
#     # assert torch.isfinite(loss).all()
#     return loss

# def compute_loss(gt_bitmasks, mask_logits):
#     # assert torch.isfinite(gt_bitmasks).all() and torch.isfinite(mask_logits).all()
#     gt_bitmasks = gt_bitmasks.float()
#     eps = 1e-5
#     intersection = (mask_logits * gt_bitmasks).sum(dim=1)
#     union = (mask_logits ** 2.0).sum(dim=1) + (gt_bitmasks ** 2.0).sum(dim=1) + eps
#     loss = 1. - (2 * intersection / union)
#     return loss.mean()

# def compute_loss(gt_bitmasks, mask_logits):
#     # assert torch.isfinite(gt_bitmasks).all() and torch.isfinite(mask_logits).all()
#     gt_bitmasks = gt_bitmasks.float()
#     num_gpus = get_world_size()

#     true_point = gt_bitmasks > 0.5
#     num_true = torch.where(true_point)[0].size(0)
#     if num_true == 0:
#         num_true = reduce_sum(mask_logits.new_tensor([num_true])).item()
#         loss1 = mask_logits.sum() * 0.0
#     else:
#         num_true = reduce_sum(mask_logits.new_tensor([num_true])).item()
#         num_true = max(num_true / num_gpus, 1.0)
#         loss1 = F.mse_loss(mask_logits[true_point], 
#                 gt_bitmasks[true_point], reduction='sum') / num_true

#     positive_point = mask_logits > 0.5
#     false_positive_point = (~true_point) | positive_point
#     num_false_positive = torch.where(false_positive_point)[0].size(0)
#     if num_false_positive == 0:
#         num_false_positive = reduce_sum(mask_logits.new_tensor([num_false_positive])).item()
#         loss2 = mask_logits.sum() * 0.0
#     else:
#         num_false_positive = reduce_sum(mask_logits.new_tensor([num_false_positive])).item()
#         num_false_positive = max(num_false_positive / num_gpus, 1.0)
#         loss2 = F.mse_loss(mask_logits[false_positive_point], 
#                 gt_bitmasks[false_positive_point], reduction='sum') / num_false_positive

#     loss = 0.5*loss1 + 0.5*loss2

#     return loss
