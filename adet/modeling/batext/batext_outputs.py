import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from adet.utils.comm import get_world_size
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
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


def fcos_losses(
        labels,
        reg_targets,
        bezier_targets,
        logits_pred,
        reg_pred,
        bezier_pred,
        ctrness_pred,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,
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
        alpha=focal_loss_alpha,
        gamma=focal_loss_gamma,
        reduction="sum",
    ) / num_pos_avg

    reg_pred = reg_pred[pos_inds]
    bezier_pred = bezier_pred[pos_inds]
    reg_targets = reg_targets[pos_inds]
    bezier_targets = bezier_targets[pos_inds]
    ctrness_pred = ctrness_pred[pos_inds]

    ctrness_targets = compute_ctrness_targets(reg_targets)
    ctrness_targets_sum = ctrness_targets.sum()
    loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

    if pos_inds.numel() > 0:
        reg_loss = iou_loss(
            reg_pred,
            reg_targets,
            ctrness_targets
        ) / loss_denorm
        
        ctrness_loss = F.binary_cross_entropy_with_logits(
            ctrness_pred,
            ctrness_targets,
            reduction="sum"
        ) / num_pos_avg
    else:
        reg_loss = reg_pred.sum() * 0
        bezier_loss = bezier_pred.sum() * 0
        ctrness_loss = ctrness_pred.sum() * 0

    bezier_loss = F.smooth_l1_loss(
        bezier_pred, bezier_targets, reduction="none")
    bezier_loss = ((bezier_loss.mean(dim=-1) * ctrness_targets).sum()
                    / loss_denorm)

    losses = {
        "loss_fcos_cls": class_loss,
        "loss_fcos_loc": reg_loss,
        "loss_fcos_ctr": ctrness_loss,
        "loss_fcos_bezier": bezier_loss,
    }
    return losses


class BATextOutputs(object):
    def __init__(
            self,
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            bezier_pred,
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
    ):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.bezier_pred = bezier_pred
        self.ctrness_pred = ctrness_pred
        self.locations = locations

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

    def _transpose(self, training_targets, num_loc_list):
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

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        bezier_targets = training_targets["bezier_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
            bezier_targets[l] = bezier_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
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
        bezier_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                bezier_targets.append(locations.new_zeros((locations.size(0), 16)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # bezier points are relative distances from center to control points
            bezier_pts = targets_per_im.beziers.view(-1, 8, 2)
            x_targets = bezier_pts[:, :, 0][None] - xs[:, None, None]
            y_targets = bezier_pts[:, :, 1][None] - ys[:, None, None]
            bezier_targets_per_im = torch.stack((x_targets, y_targets), dim=3)
            bezier_targets_per_im = bezier_targets_per_im.view(xs.size(0), bboxes.size(0), 16)

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
            bezier_targets_per_im = bezier_targets_per_im[range(len(locations)), locations_to_gt_inds]
            # target_inds_per_im = locations_to_gt_inds + num_targets

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            bezier_targets.append(bezier_targets_per_im)
            # target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "bezier_targets": bezier_targets}

    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets, bezier_targets = (
            training_targets["labels"],
            training_targets["reg_targets"],
            training_targets["bezier_targets"])

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
        bezier_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 16)
                for x in self.bezier_pred
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

        bezier_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 16) -> (N*Hi*Wi, 16)
                x.reshape(-1, 16) for x in bezier_targets
            ], dim=0,)

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0,)

        return fcos_losses(
            labels,
            reg_targets,
            bezier_targets,
            logits_pred,
            reg_pred,
            bezier_pred,
            ctrness_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
        )

    def predict_proposals(self, top_feats):
        sampled_boxes = []

        bundle = {
            "l": self.locations, "o": self.logits_pred,
            "r": self.reg_pred, "c": self.ctrness_pred,
            "s": self.strides,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, instance in enumerate(zip(*bundle.values())):
            instance_dict = dict(zip(bundle.keys(), instance))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = instance_dict["l"]
            o = instance_dict["o"]
            r = instance_dict["r"] * instance_dict["s"]
            c = instance_dict["c"]
            # top_feat is the bezier regression
            t = instance_dict["t"] * instance_dict["s"] if "t" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, self.image_sizes, t
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness,
            image_sizes, top_feat=None):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()
        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

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
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

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
            if top_feat is not None:
                # top_feat in batext is bezier regression
                # decentralize
                bezier_detections = per_locations.unsqueeze(1) + per_top_feat.view(-1, 8, 2) 
                bezier_detections = bezier_detections.view(-1, 16)
                boxlist.top_feat = bezier_detections
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
