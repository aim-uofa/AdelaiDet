# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_MEInst_config(cfg):
    """
    Add config for MEInst: https://arxiv.org/pdf/2003.11712
    """
    cfg.MODEL.MEInst = CN()

    # ---------------------------------------------------------------------------- #
    # MEInst Head
    # ---------------------------------------------------------------------------- #

    # This is the number of foreground classes.
    cfg.MODEL.MEInst.NUM_CLASSES = 80
    cfg.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.MEInst.PRIOR_PROB = 0.01
    cfg.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.MEInst.NMS_TH = 0.6
    cfg.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.MEInst.TOP_LEVELS = 2
    cfg.MODEL.MEInst.NORM = "GN"  # Support GN or none
    cfg.MODEL.MEInst.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    cfg.MODEL.MEInst.THRESH_WITH_CTR = False

    # Focal loss parameters
    cfg.MODEL.MEInst.LOSS_ALPHA = 0.25
    cfg.MODEL.MEInst.LOSS_GAMMA = 2.0
    cfg.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.MEInst.USE_RELU = True
    cfg.MODEL.MEInst.USE_DEFORMABLE = False
    cfg.MODEL.MEInst.LAST_DEFORMABLE = False
    cfg.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.MEInst.NUM_CLS_CONVS = 4
    cfg.MODEL.MEInst.NUM_BOX_CONVS = 4
    cfg.MODEL.MEInst.NUM_SHARE_CONVS = 0
    cfg.MODEL.MEInst.CENTER_SAMPLE = True
    cfg.MODEL.MEInst.POS_RADIUS = 1.5
    cfg.MODEL.MEInst.LOC_LOSS_TYPE = 'giou'

    # ---------------------------------------------------------------------------- #
    # Mask Encoding
    # ---------------------------------------------------------------------------- #
    # Whether to use mask branch.
    cfg.MODEL.MEInst.MASK_ON = True
    # IOU overlap ratios [IOU_THRESHOLD]
    # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
    # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
    cfg.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.MEInst.IOU_LABELS = [0, 1]
    # Whether to use class_agnostic or class_specific.
    cfg.MODEL.MEInst.AGNOSTIC = True
    # Some operations in mask encoding.
    cfg.MODEL.MEInst.WHITEN = True
    cfg.MODEL.MEInst.SIGMOID = True

    # The number of convolutions used in the mask tower.
    cfg.MODEL.MEInst.NUM_MASK_CONVS = 4

    # The dim of mask before/after mask encoding.
    cfg.MODEL.MEInst.DIM_MASK = 60
    cfg.MODEL.MEInst.MASK_SIZE = 28
    # The default path for parameters of mask encoding.
    cfg.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                       "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
    # An indicator for encoding parameters loading during training.
    cfg.MODEL.MEInst.FLAG_PARAMETERS = False
    # The loss for mask branch, can be mse now.
    cfg.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

    # Whether to use gcn in mask prediction.
    # Large Kernel Matters -- https://arxiv.org/abs/1703.02719
    cfg.MODEL.MEInst.USE_GCN_IN_MASK = False
    cfg.MODEL.MEInst.GCN_KERNEL_SIZE = 9
    # Whether to compute loss on original mask (binary mask).
    cfg.MODEL.MEInst.LOSS_ON_MASK = False


