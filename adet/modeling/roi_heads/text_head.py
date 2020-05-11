import math

from typing import Dict, List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY

from adet.layers import conv_with_kaiming_uniform
from ..poolers import TopPooler
from .attn_predictor import ATTPredictor


class SeqConvs(nn.Module):
    def __init__(self, conv_dim, roi_size):
        super().__init__()

        height = roi_size[0]
        downsample_level = math.log2(height) - 2
        assert math.isclose(downsample_level, int(downsample_level))
        downsample_level = int(downsample_level)

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        convs = []
        for i in range(downsample_level):
            convs.append(conv_block(
                conv_dim, conv_dim, 3, stride=(2, 1)))
        convs.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=(4, 1), bias=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class RNNPredictor(nn.Module):
    def __init__(self, cfg):
        super(RNNPredictor, self).__init__()
        # fmt: off
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        conv_dim          = cfg.MODEL.BATEXT.CONV_DIM
        roi_size          = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        # fmt: on

        self.convs = SeqConvs(conv_dim, roi_size)
        self.rnn = nn.LSTM(conv_dim, conv_dim, num_layers=1, bidirectional=True)
        self.clf = nn.Linear(conv_dim * 2, self.voc_size + 1)
        self.recognition_loss_fn = build_recognition_loss_fn()

    def forward(self, x, targets=None):
        # check empty
        if x.size(0) == 0:
            return x.new_zeros((x.size(2), 0, self.voc_size))
        x = self.convs(x).squeeze(dim=2)  # NxCxW
        x = x.permute(2, 0, 1)  # WxNxC
        x, _ = self.rnn(x)
        preds = self.clf(x)

        if self.training:
            rec_loss = self.recognition_loss_fn(preds, targets, self.voc_size)
            return preds, rec_loss
        else:
            # (W, N, C) -> (N, W, C)
            _, preds = preds.permute(1, 0, 2).max(dim=-1)
            return preds, None


def build_recognizer(cfg, type):
    if type == 'rnn':
        return RNNPredictor(cfg)
    if type == 'attn':
        return ATTPredictor(cfg)
    else:
        raise NotImplementedError("{} is not a valid recognizer".format(type))


def ctc_loss(preds, targets, voc_size):
    # prepare targets
    target_lengths = (targets != voc_size).long().sum(dim=-1)
    trimmed_targets = [t[:l] for t, l in zip(targets, target_lengths)]
    targets = torch.cat(trimmed_targets)

    x = F.log_softmax(preds, dim=-1)
    input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
    return F.ctc_loss(
        x, targets, input_lengths, target_lengths,
        blank=voc_size
    )


def build_recognition_loss_fn(rec_type="ctc"):
    if rec_type == "ctc":
        return ctc_loss
    else:
        raise NotImplementedError("{} is not a valid recognition loss".format(rec_type))


@ROI_HEADS_REGISTRY.register()
class TextHead(nn.Module):
    """
    TextHead performs text region alignment and recognition.
    
    It is a simplified ROIHeads, only ground truth RoIs are
    used during training.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super(TextHead, self).__init__()
        # fmt: off
        pooler_resolution = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        pooler_scales     = cfg.MODEL.BATEXT.POOLER_SCALES
        sampling_ratio    = cfg.MODEL.BATEXT.SAMPLING_RATIO
        conv_dim          = cfg.MODEL.BATEXT.CONV_DIM
        num_conv          = cfg.MODEL.BATEXT.NUM_CONV
        canonical_size    = cfg.MODEL.BATEXT.CANONICAL_SIZE
        self.in_features  = cfg.MODEL.BATEXT.IN_FEATURES
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        recognizer        = cfg.MODEL.BATEXT.RECOGNIZER
        self.top_size     = cfg.MODEL.TOP_MODULE.DIM
        # fmt: on

        self.pooler = TopPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="BezierAlign",
            canonical_box_size=canonical_size,
            canonical_level=3,
            assign_crit="bezier")

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        tower = []
        for i in range(num_conv):
            tower.append(
                conv_block(conv_dim, conv_dim, 3, 1))
        self.tower = nn.Sequential(*tower)
        
        self.recognizer = build_recognizer(cfg, recognizer)

    def forward(self, images, features, proposals, targets=None):
        """
        see detectron2.modeling.ROIHeads
        """
        del images

        features = [features[f] for f in self.in_features]
        if self.training:
            beziers = [p.beziers for p in targets]
            targets = torch.cat([x.text for x in targets], dim=0)
        else:
            beziers = [p.top_feat for p in proposals]
        bezier_features = self.pooler(features, beziers)
        bezier_features = self.tower(bezier_features)

        # TODO: move this part to recognizer
        if self.training:
            preds, rec_loss = self.recognizer(bezier_features, targets)
            rec_loss *= 0.05
            losses = {'rec_loss': rec_loss}
            return None, losses
        else:
            if bezier_features.size(0) == 0:
                for box in proposals:
                    box.beziers = box.top_feat
                    box.recs = box.top_feat
                return proposals, {}
            preds, _ = self.recognizer(bezier_features, targets)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.recs = preds[start_ind:end_ind]
                proposals_per_im.beziers = proposals_per_im.top_feat
                start_ind = end_ind

            return proposals, {}