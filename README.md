# AdelaiDet

AdelaiDet is an open source toolbox for multiple instance-level recognition tasks on top of [Detectron2](https://github.com/facebookresearch/detectron2).
All instance-level recognition works from our group are open-sourced here.

To date, AdelaiDet implements the following algorithms:

* [FCOS](https://arxiv.org/abs/1904.01355)
* [BlendMask](https://arxiv.org/abs/2001.00309) _to be released_
* [ABCNet](https://arxiv.org/abs/2002.10200) _to be released_ ([demo](https://github.com/Yuliang-Liu/bezier_curve_text_spotting))
* [SOLO](https://arxiv.org/abs/1912.04488) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [SOLOv2](https://arxiv.org/abs/2003.10152) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [DirectPose](https://arxiv.org/abs/1911.07451) _to be released_
* [CondInst](https://arxiv.org/abs/2003.05664) _to be released_


## Models

More models will be released soon. Stay tuned.

### COCO Object Detecton Baselines with FCOS

Name | box AP | download
--- |:---:|:---:
[FCOS_R_50_1x](configs/FCOS-Detection/R_50_1x.yaml) | 38.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download)

### COCO Instance Segmentation Baselines with [BlendMask](https://arxiv.org/abs/2001.00309)

Model | Name |inference time (ms/im) | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:
Mask R-CNN | [550_R_50_3x](configs/RCNN/550_R_50_FPN_3x.yaml) | 63 | 39.1 | 35.3 |
BlendMask | [550_R_50_3x](configs/BlendMask/550_R_50_3x.yaml) | 40 | 38.7 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/o0bpkmhMiuYgIcQ/download)
Mask R-CNN | [R_50_1x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 90 | 38.6 | 35.2 |
BlendMask | [R_50_1x](configs/BlendMask/R_50_1x.yaml) | 83 | 39.9 | 35.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/crpmeVCnQ3StvSz/download)
Mask R-CNN | [R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) |  | 41.0 | 37.2 | 
BlendMask | [R_50_3x](configs/BlendMask/R_50_3x.yaml) |  | 42.7 | 37.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/9u1cG2zXvEva5SM/download)
Mask R-CNN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) |  | 42.9 | 38.6 |
BlendMask | [R_101_3x](configs/BlendMask/R_101_3x.yaml) |  | 44.8 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/mYm5VCXICoeLNHq/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/R_101_dcni3_5x.yaml) |  | 46.8 | 41.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/TAZPxSDvPuhegKp/download)

### COCO Panoptic Segmentation Baselines with BlendMask
Model | Name | PQ | PQ<sup>Th</sup> | PQ<sup>St</sup> | download
--- |:---:|:---:|:---:|:---:|:---:
Panoptic FPN | [R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml) | 41.5 | 48.3 | 31.2 | 
BlendMask | [R_50_3x](configs/BlendMask/Panoptic/R_50_3x.yaml) | 42.5 | 49.5 | 32.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/bG0IhYeMAvlTGTq/download)
Panoptic FPN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/panoptic_fpn_R_101_3x.yaml) | 43.0 | 49.7 | 32.9 |
BlendMask | [R_101_3x](configs/BlendMask/Panoptic/R_101_3x.yaml) | 44.3 | 51.6 | 33.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/AEwbhyQ9F3lqvsz/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/Panoptic/R_101_dcni3_5x.yaml) | 46.0 | 52.9 | 35.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/GyWDhsukAYYokZg/download)

### Person in Context with BlendMask
Model | Name | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
BlendMask | [R_50_1x](configs/BlendMask/Person/R_50_1x.yaml) | 70.6 | 66.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/d4f16WshXYbOuIo)

## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Then build AdelaiDet with:
```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```

## Quick Start

### Inference with Pre-trained Models

1. Pick a model and its config file, for example, `fcos_R_50_1x.yaml`.
2. Download the model `wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth`
3. Run the demo with
```
python demo/demo.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --input input1.jpg input2.jpg \
	--opts MODEL.WEIGHTS fcos_R_50_1x.pth
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
python tools/train_net.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/fcos_R_50_1x
```
To evaluate the model after training, run:

```
python tools/train_net.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/fcos_R_50_1x \
    MODEL.WEIGHTS training_dir/fcos_R_50_1x/model_final.pth
```

The configs are made for 8-GPU training. To train on another number of GPUs, change the `num-gpus`.


## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

```BibTeX
@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}

@inproceedings{chen2020blendmask,
  title     =  {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@article{wang2019solo,
  title   =  {{SOLO}: Segmenting Objects by Locations},
  author  =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  journal =  {arXiv preprint arXiv:1912.04488},
  year    =  {2019}
}

@article{wang2020solov2,
  title   =  {{SOLOv2}: Dynamic, Faster and Stronger},
  author  =  {Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:2003.10152},
  year    =  {2020}
}

@article{tian2019directpose,
  title   =  {{DirectPose}: Direct End-to-End Multi-Person Pose Estimation},
  author  =  {Tian, Zhi and Chen, Hao and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1911.07451},
  year    =  {2019}
}

@article{tian2020conditional,
  title   = {Conditional Convolutions for Instance Segmentation},
  author  = {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  journal = {arXiv preprint arXiv:2003.05664},
  year    = {2020}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/).
