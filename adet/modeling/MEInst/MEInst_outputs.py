import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes, pairwise_iou

from detectron2.utils.comm import get_world_size
from detectron2.modeling.matcher import Matcher

from fvcore.nn import sigmoid_focal_loss_jit

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores
    
    mask_regression: the predicted mask coefficients (D)
    
"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class MEInstOutputs(object):
    def __init__(
            self,
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            mask_regression,
            mask_encoding,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            gt_instances=None,
            cfg=None,
    ):
        self.cfg = cfg
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.ctrness_pred = ctrness_pred
        self.locations = locations
        self.mask_regression = mask_regression
        self.mask_encoding = mask_encoding

        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr

        self.loss_on_mask = cfg.MODEL.MEInst.LOSS_ON_MASK
        self.mask_loss_type = cfg.MODEL.MEInst.MASK_LOSS_TYPE
        self.dim_mask = cfg.MODEL.MEInst.DIM_MASK
        self.mask_size = cfg.MODEL.MEInst.MASK_SIZE
        if self.loss_on_mask:
            self.mask_loss_func = nn.BCEWithLogitsLoss(reduction="none")
        elif self.mask_loss_type == 'mse':
            self.mask_loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.MEInst.IOU_THRESHOLDS,
            cfg.MODEL.MEInst.IOU_LABELS,
            allow_low_quality_matches=False,
        )

    @torch.no_grad()
    def prepare_masks(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # No proposal boxes available for images during training.
            if not len(proposals_per_image):
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((0, 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                proposals_with_gt.append(proposals_per_image)
                continue

            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.pos_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[matched_idxs])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(matched_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    @staticmethod
    def _transpose(training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, self.gt_instances, loc_to_size_range
        )

        # Mask Encoding
        mask_targets = training_targets.pop("mask_targets")
        mask_indices = training_targets.pop("mask_indices")
        mask_targets = self.prepare_masks(mask_targets, self.gt_instances)
        mask_targets_split = []
        for mask_target_per_img, mask_index_per_img in zip(mask_targets, mask_indices):
            assert len(mask_target_per_img) == len(mask_index_per_img), print(
                "The number(positive) should be equal between mask_target and mask_index, "
                "which means that the function(prepare_masks) should not filter any proposals, "
                "the mask should be generated one by one according to the input proposals.")
            # there is no gt target assigned to the image.
            if len(mask_target_per_img) == 0:
                continue
            mask_level = []
            level_s = 0
            for level_e in num_loc_list:
                level_e += level_s
                level_ge = mask_index_per_img.ge(level_s)
                level_lt = mask_index_per_img.lt(level_e)
                index_level = torch.nonzero(level_ge * level_lt).squeeze(1)
                mask_target_per_level = mask_target_per_img[index_level].gt_masks.crop_and_resize(
                                        mask_target_per_img[index_level].pos_boxes.tensor,
                                        self.mask_size).float()
                mask_level.append(mask_target_per_level)
                level_s = level_e
            mask_targets_split.append(mask_level)

        mask_targets_level_first = []
        for level in range(len(self.locations)):
            mask_targets_level_first.append(
                torch.cat([mask_per_im[level] for mask_per_im in mask_targets_split], dim=0)
            )

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        labels_level_first = training_targets["labels"]
        for labels_per_level, mask_targets_level in zip(labels_level_first, mask_targets_level_first):
            num_pos = (labels_per_level != self.num_classes).nonzero().numel()
            assert num_pos == mask_targets_level.shape[0], \
                print("The number(positive) should be equal between labels_per_level and mask_targets_level.")

        # append mask_targets to training targets.
        training_targets["mask_targets"] = mask_targets_level_first

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    @staticmethod
    def get_sample_region(gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        mask_targets = []
        mask_indices = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

            # Mask Encoding.
            pos_inds = torch.nonzero(labels_per_im != self.num_classes).squeeze(1)
            pos_labels = labels_per_im[pos_inds]
            pos_reg_targets = reg_targets_per_im[pos_inds]
            pos_locations = locations[pos_inds]
            bbs = torch.stack([
                pos_locations[:, 0] - pos_reg_targets[:, 0],
                pos_locations[:, 1] - pos_reg_targets[:, 1],
                pos_locations[:, 0] + pos_reg_targets[:, 2],
                pos_locations[:, 1] + pos_reg_targets[:, 3],
            ], dim=1)
            bbs = Boxes(bbs)

            mask_targets_per_im = Instances(targets_per_im.image_size)
            mask_targets_per_im.set("pos_classes", pos_labels)
            mask_targets_per_im.set("pos_boxes", bbs)

            mask_targets.append(mask_targets_per_im)
            mask_indices.append(pos_inds)

        return {"labels": labels, "reg_targets": reg_targets,
                "mask_targets": mask_targets, "mask_indices": mask_indices}

    def MEInst_losses(
            self,
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            ctrness_pred,
            mask_pred,
            mask_targets
    ):
        num_classes = logits_pred.size(1)
        labels = labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        reg_pred = reg_pred[pos_inds]
        reg_targets = reg_targets[pos_inds]
        ctrness_pred = ctrness_pred[pos_inds]
        mask_pred = mask_pred[pos_inds]
        assert mask_pred.shape[0] == mask_targets.shape[0], \
            print("The number(positive) should be equal between "
                  "masks_pred(prediction) and mask_targets(target).")

        ctrness_targets = compute_ctrness_targets(reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        reg_loss = self.iou_loss(
            reg_pred,
            reg_targets,
            ctrness_targets
        ) / ctrness_norm

        ctrness_loss = F.binary_cross_entropy_with_logits(
            ctrness_pred,
            ctrness_targets,
            reduction="sum"
        ) / num_pos_avg

        if self.loss_on_mask:
            # n_components predictions --> m*m mask predictions without sigmoid
            # as sigmoid function is combined in loss.
            mask_pred = self.mask_encoding.decoder(mask_pred, is_train=True)
            mask_loss = self.mask_loss_func(
                mask_pred,
                mask_targets
            )
            mask_loss = mask_loss.sum(1) * ctrness_targets
            mask_loss = mask_loss.sum() / max(ctrness_norm * self.mask_size ** 2, 1.0)
        else:
            # m*m mask labels --> n_components encoding labels
            mask_targets = self.mask_encoding.encoder(mask_targets)
            if self.mask_loss_type == 'mse':
                mask_loss = self.mask_loss_func(
                    mask_pred,
                    mask_targets
                )
                mask_loss = mask_loss.sum(1) * ctrness_targets
                mask_loss = mask_loss.sum() / max(ctrness_norm * self.dim_mask, 1.0)
            else:
                raise NotImplementedError

        losses = {
            "loss_MEInst_cls": class_loss,
            "loss_MEInst_loc": reg_loss,
            "loss_MEInst_ctr": ctrness_loss,
            "loss_MEInst_mask": mask_loss,
        }
        return losses, {}

    def losses(self):
        """
        Return the losses from a set of MEInst predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets, mask_targets = training_targets["labels"], training_targets["reg_targets"], \
                                            training_targets["mask_targets"]

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.logits_pred
            ], dim=0,)
        reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.reg_pred
            ], dim=0,)
        ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.ctrness_pred
            ], dim=0,)

        labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels
            ], dim=0,)

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0,)

        mask_pred = cat(
            [
                # Reshape: (N, D, Hi, Wi) -> (N, Hi, Wi, D) -> (N*Hi*Wi, D)
                x.permute(0, 2, 3, 1).reshape(-1, self.dim_mask)
                for x in self.mask_regression
            ], dim=0,)

        mask_targets = cat(
            [
                # Reshape: (N, Hi, Wi, mask_size^2) -> (N*Hi*Wi, mask_size^2)
                x.reshape(-1, self.mask_size ** 2) for x in mask_targets
            ], dim=0,)

        return self.MEInst_losses(
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            ctrness_pred,
            mask_pred,
            mask_targets
        )

    def predict_proposals(self):
        sampled_boxes = []

        bundle = (
            self.locations, self.logits_pred,
            self.reg_pred, self.ctrness_pred,
            self.strides, self.mask_regression
        )

        for i, (l, o, r, c, s, mr) in enumerate(zip(*bundle)):
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            r = r * s
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, mr, self.image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        num_images = len(boxlists)
        for i in range(num_images):
            per_image_masks = boxlists[i].pred_masks
            per_image_masks = self.mask_encoding.decoder(per_image_masks, is_train=False)
            per_image_masks = per_image_masks.view(-1, 1, self.mask_size, self.mask_size)
            boxlists[i].pred_masks = per_image_masks

        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness, mask_regression, image_sizes
    ):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()
        mask_regression = mask_regression.view(N, self.dim_mask, H, W).permute(0, 2, 3, 1)
        mask_regression = mask_regression.reshape(N, -1, self.dim_mask)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_box_mask = mask_regression[i]
            per_box_mask = per_box_mask[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_box_mask = per_box_mask[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.pred_masks = per_box_mask

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
