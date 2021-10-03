import torch
from torch.nn import functional as F
from torch import nn

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler
from .utils import aligned_bilinear, compute_loss, compute_loss_softmax
from fvcore.nn import sigmoid_focal_loss_jit
from adet.utils.comm import reduce_sum
from detectron2.utils.comm import get_world_size
from detectron2.layers import ConvTranspose2d
from detectron2.structures.instances import Instances

import logging


logger = logging.getLogger("detectron2.blender")


def build_blender(cfg):
    return Blender(cfg)


def compute_locations_per_level(h, w, stride, device):
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


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def get_subnetworks_params(attns, num_bases, channels):
    assert attns.dim() == 2
    n_inst = attns.size(0)

    w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
        (2 + num_bases) * channels, channels,
        channels * channels, channels,
        channels * 17, 17
    ], dim=1)

    # out_channels x in_channels x 1 x 1
    w0 = w0.reshape(n_inst * channels, 2 + num_bases, 1, 1)
    b0 = b0.reshape(n_inst * channels)
    w1 = w1.reshape(n_inst * channels, channels, 1, 1)
    b1 = b1.reshape(n_inst * channels)
    w2 = w2.reshape(n_inst * 17, channels, 1, 1)
    b2 = b2.reshape(n_inst*17)

    return [w0, w1, w2], [b0, b1, b2]


def subnetworks_forward(inputs, weights, biases, n_subnets):
    '''
    :param inputs: a list of inputs
    :param weights: [w0, w1, ...]
    :param bias: [b0, b1, ...]
    :return:
    '''
    assert inputs.dim() == 4
    n_layer = len(weights)
    x = inputs
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = F.conv2d(
            x, w, bias=b,
            stride=1, padding=0,
            groups=n_subnets
        )
        if i < n_layer - 1:
            x = F.relu(x)
    return x


class fcpose_head_module(nn.Module):
    def __init__(self, cfg):
        super(fcpose_head_module, self).__init__()
        # fmt: off

        self.attn_len             = cfg.MODEL.FCPOSE.ATTN_LEN
        self.dynamic_channels     = cfg.MODEL.FCPOSE.DYNAMIC_CHANNELS
        self.max_proposals_per_im = cfg.MODEL.FCPOSE.MAX_PROPOSALS
        self.loss_weight_keypoint = cfg.MODEL.FCPOSE.LOSS_WEIGHT_KEYPOINT
        self.distance_norm        = cfg.MODEL.FCPOSE.DISTANCE_NORM
        self.sizes_of_interest    = nn.Parameter(torch.tensor(
                                    cfg.MODEL.FCOS.SIZES_OF_INTEREST + [cfg.INPUT.MAX_SIZE_TRAIN]
                                    ), requires_grad=False)
        self.device                = cfg.MODEL.DEVICE
        self.focal_loss_alpha      = cfg.MODEL.FCPOSE.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma      = cfg.MODEL.FCPOSE.FOCAL_LOSS_GAMMA
        self.atten_producer        = nn.Linear(256*9, self.attn_len)
        self.upsampler             = nn.Upsample(scale_factor = 4, mode = 'bilinear')
        self.loss_weight_direction = cfg.MODEL.FCPOSE.LOSS_WEIGHT_DIRECTION
        self.mask_dim              = cfg.MODEL.FCPOSE.BASIS_MODULE.NUM_BASES
        # fmt: on

    def select_pred_inst(self, proposals):
        pred_instances           = proposals["instances"]
        N                        = len(pred_instances.image_size)
        try:
            num_instance             = pred_instances.gt_inds.max() + 1
        except: #TODO fix this bug
            print('error')
            print(pred_instances)
            num_instance = 0
        max_num_instances_per_gt = self.max_proposals_per_im // num_instance
        max_num_instances_per_gt = max(max_num_instances_per_gt,1)

        kept_instances = []
        num_loss       = []
        for im_id in range(N):
            instances_per_im = pred_instances[pred_instances.im_inds == im_id]
            if len(instances_per_im) == 0:
                continue

            unique_gt_inds = instances_per_im.gt_inds.unique()

            for gt_ind in unique_gt_inds:
                instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                if len(instances_per_gt) > max_num_instances_per_gt:
                    scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                    ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                    inds = (scores * ctrness_pred).topk(k=max_num_instances_per_gt, dim=0)[1]
                    instances_per_gt = instances_per_gt[inds]
                    num_loss_per_inst = unique_gt_inds.new_full([max_num_instances_per_gt], max_num_instances_per_gt)
                else:
                    num_loss_per_inst = unique_gt_inds.new_full([len(instances_per_gt)], len(instances_per_gt))


                kept_instances.append(instances_per_gt)
                num_loss.append(num_loss_per_inst)

        pred_instances = Instances.cat(kept_instances)
        num_loss = torch.cat(num_loss,dim=0)
        # del kept_instances, proposals

        # pred_instances.mask_head_params = pred_instances.top_feats
        attns = pred_instances.top_feats
        im_inds = pred_instances.im_inds
        locations = pred_instances.locations
        levels = pred_instances.fpn_levels
        gt_inds = pred_instances.gt_inds

        return attns, im_inds, locations, levels, gt_inds, num_instance, num_loss

    def forward(self, bases, proposals, head_gt_heatmap, gt_instances, basis_seg, base_stride = 8):
        bases = bases[0]
        bases, direction = torch.split_with_sizes(bases, [self.mask_dim,34], dim=1)
        base_locations = compute_locations_per_level(
                        bases.size(2), bases.size(3),
                        stride=base_stride, device=bases.device
                        )

        N,num_bases,H,W = bases.shape
        direction_locations = base_locations.reshape(H,W,2)
        direction_locations = direction_locations.permute(2,0,1)[None].repeat(N,17,1,1) # N, 17*2, H ,W
        direction = direction + direction_locations

        if self.training:
            
            attns, im_inds, locations, levels, gt_inds, num_instance, num_loss = \
                self.select_pred_inst(proposals)

            gt_keypoint = []
            gt_box      = []
            for per_keypoint in gt_instances:
                gt_keypoint.append(per_keypoint.gt_keypoints.tensor)
                gt_box.append(per_keypoint.gt_boxes.tensor)
            gt_keypoint = torch.cat(gt_keypoint,dim=0)[gt_inds]
            gt_box      = torch.cat(gt_box,dim=0)[gt_inds]
            gt_bitmasks = head_gt_heatmap[gt_inds]
            direction   = direction[im_inds]
            max_ranges  = self.sizes_of_interest[levels.long()]

            n_inst = attns.size(0)


            assert not torch.isnan(locations).any()
            assert not torch.isnan(base_locations).any()
            assert not torch.isnan(bases).any()
            offsets = locations.reshape(-1, 1, 2) - base_locations.reshape(1, -1, 2)
            offsets = offsets.permute(0, 2, 1).float() / max_ranges.reshape(-1, 1, 1).float()
            offsets = torch.cat([offsets, bases[im_inds].reshape(n_inst, num_bases, -1)], dim=1)
            offsets = offsets.reshape(1, -1, H, W)

            assert not torch.isnan(attns).any()
            attns = self.atten_producer(attns)
            weights, biases = get_subnetworks_params(attns, num_bases, self.dynamic_channels)
            for weight, biase in zip(weights,biases):
                assert not torch.isnan(weight).any()
                assert not torch.isnan(biase).any()
            assert not torch.isnan(offsets).any()

            mask_logits = subnetworks_forward(offsets, weights, biases, n_inst).squeeze()

            mask_logits = mask_logits.reshape(-1, 17, H, W)
            larger_mask_logits = self.upsampler(mask_logits)
            assert not torch.isnan(larger_mask_logits).any()

            assert not torch.isnan(mask_logits).any()
            mask_logits = mask_logits.flatten(start_dim=2).softmax(dim=2).reshape(-1, 17, H, W)
            direction = direction[:,:,:,:,None].permute(0,2,3,4,1).reshape(n_inst,H,W,17,2)
            mask_logits = mask_logits.permute(0,2,3,1)[:,:,:,:,None]

            del weights, biases


            gt_box_x = gt_box[:,2] - gt_box[:,0]
            gt_box_y = gt_box[:,3] - gt_box[:,1]
            max_ranges = (gt_box_x + gt_box_y) / 2
            keypoint_loss, direction_loss = \
                compute_loss_softmax(gt_bitmasks, larger_mask_logits,
                                num_loss, num_instance, direction, mask_logits, gt_keypoint, 
                                max_ranges, self.distance_norm)


            return None, {"loss_keypoint": keypoint_loss * self.loss_weight_keypoint,
                            "loss_direction": direction_loss * self.loss_weight_direction}
        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_keypoints = box.pred_classes.new_full((0,17,3),0).float()
                return proposals, {}

            N, num_bases, H, W = bases.size()
            for im_i in range(len(proposals)):
                per_attns = proposals[im_i].top_feat
                n_inst = per_attns.size(0)

                per_locations = proposals[im_i].locations
                max_ranges = self.sizes_of_interest[proposals[im_i].fpn_levels.long()]

                offsets = per_locations.reshape(-1, 1, 2) - base_locations.reshape(1, -1, 2)
                offsets = offsets.permute(0, 2, 1).float() / max_ranges.reshape(-1, 1, 1).float()
                # offsets = offsets.tanh()
                offsets = torch.cat([offsets, bases[im_i].reshape(1, num_bases, -1).expand(n_inst, -1, -1)], dim=1)
                offsets = offsets.reshape(1, -1, H, W)

                attns = self.atten_producer(per_attns)
                weights, biases = get_subnetworks_params(attns, num_bases, self.dynamic_channels)
                pred_mask_logits = subnetworks_forward(offsets, weights, biases, n_inst)

                pred_mask_logits = pred_mask_logits.reshape(-1, 17, H, W)
                # pred_mask_logits = pred_mask_logits.flatten(start_dim=2).softmax(dim=2).reshape(-1, 17, H, W)
                direction = direction.repeat(n_inst,1,1,1)
                direction = direction[:,:,:,:,None].permute(0,2,3,4,1).reshape(n_inst,H,W,17,2)
                pred_mask_logits = pred_mask_logits.permute(0,2,3,1)[:,:,:,:,None]

                pred_mask_logits = pred_mask_logits.reshape(n_inst,H*W,17).permute(0,2,1)#.sigmoid()
                pred_mask_logits = pred_mask_logits.reshape(n_inst*17,H*W)
                max_value, max_index = pred_mask_logits.max(dim = 1)
                arr = torch.arange(n_inst*17, device=pred_mask_logits.device)
                direction = direction.permute(0,3,1,2,4).reshape(n_inst*17,H*W,2)
                direction = direction[arr,max_index]
                pred_keypoints = direction.reshape(n_inst,17,2)

                vis = max_value.reshape(pred_keypoints.size(0),pred_keypoints.size(1),1)
                # vis = pred_keypoints.new_ones((pred_keypoints.size(0), pred_keypoints.size(1), 1))
                pred_keypoints = torch.cat([pred_keypoints, vis], dim = 2)

                proposals[im_i].set("pred_keypoints", pred_keypoints)


            return proposals, {}



