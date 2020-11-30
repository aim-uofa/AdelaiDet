import torch
import torch.nn.functional as F

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()

    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def point_nms(heat, kernel=2):
    # kernel must be 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay / soft nms
    delay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'linear':
        delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
        delay_coefficient, _ = delay_matrix.min(0)
    else:
        delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = cate_scores * delay_coefficient

    return cate_scores_update


def mask_nms(cate_labels, seg_masks, sum_masks, cate_scores, nms_thr=0.5):
    n_samples = len(cate_scores)
    if n_samples == 0:
        return []

    keep = seg_masks.new_ones(cate_scores.shape)
    seg_masks = seg_masks.float()

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            label_j = cate_labels[j]
            if label_i != label_j:
                continue
            # overlaps
            inter = (mask_i * mask_j).sum()
            union = sum_masks[i] + sum_masks[j] - inter
            if union > 0:
                if inter / union > nms_thr:
                    keep[j] = False
            else:
                keep[j] = False
    return keep
