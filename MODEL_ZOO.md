# AdelaiDet Model Zoo and Baselines

## Introduction
This file documents a collection of models trained with AdelaiDet in Nov, 2019.

## Models

The inference time is measured on one 1080Ti based on the most recent commit on Detectron2 ([ffff8ac](https://github.com/facebookresearch/detectron2/commit/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9)).

More models will be released soon. Stay tuned.

### COCO Object Detecton Baselines with FCOS

Name | box AP | download
--- |:---:|:---:
[FCOS_R_50_1x](configs/FCOS-Detection/R_50_1x.yaml) | 38.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download)

### COCO Instance Segmentation Baselines with [BlendMask](https://arxiv.org/abs/2001.00309)

Model | Name |inference time (ms/im) | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:
Mask R-CNN | [550_R_50_3x](configs/RCNN/550_R_50_FPN_3x.yaml) | 63 | 39.1 | 35.3 |
BlendMask | [550_R_50_3x](configs/BlendMask/550_R_50_3x.yaml) | 36 | 38.7 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/R3Qintf7N8UCiIt/download)
Mask R-CNN | [R_50_1x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 80 | 38.6 | 35.2 |
BlendMask | [R_50_1x](configs/BlendMask/R_50_1x.yaml) | 73 | 39.9 | 35.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/zoxXPnr6Hw3OJgK/download)
Mask R-CNN | [R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) | 80 | 41.0 | 37.2 | 
BlendMask | [R_50_3x](configs/BlendMask/R_50_3x.yaml) | 74 | 42.7 | 37.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/ZnaInHFEKst6mvg/download)
Mask R-CNN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) | 100 | 42.9 | 38.6 |
BlendMask | [R_101_3x](configs/BlendMask/R_101_3x.yaml) | 94 | 44.8 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/e4fXrliAcMtyEBy/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/R_101_dcni3_5x.yaml) | 105 | 46.8 | 41.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download)

### COCO Panoptic Segmentation Baselines with BlendMask
Model | Name | PQ | PQ<sup>Th</sup> | PQ<sup>St</sup> | download
--- |:---:|:---:|:---:|:---:|:---:
Panoptic FPN | [R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml) | 41.5 | 48.3 | 31.2 | 
BlendMask | [R_50_3x](configs/BlendMask/Panoptic/R_50_3x.yaml) | 42.5 | 49.5 | 32.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/oDgi0826JOJXCr5/download)
Panoptic FPN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/panoptic_fpn_R_101_3x.yaml) | 43.0 | 49.7 | 32.9 |
BlendMask | [R_101_3x](configs/BlendMask/Panoptic/R_101_3x.yaml) | 44.3 | 51.6 | 33.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/u6gZwj06MWDEkYe/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/Panoptic/R_101_dcni3_5x.yaml) | 46.0 | 52.9 | 35.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/Jwp41WEzDdrhWsN/download)

### Person in Context with BlendMask
Model | Name | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
BlendMask | [R_50_1x](configs/BlendMask/Person/R_50_1x.yaml) | 70.6 | 66.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/nvpcKTFA5fsagc0/download)