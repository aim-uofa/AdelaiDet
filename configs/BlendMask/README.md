
# BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation

    BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation;
    Hao Chen, Kunyang Sun, Zhi Tian, Chunhua Shen, Yongming Huang, and Youliang Yan;
    In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2020.

[[`Paper`](https://arxiv.org/abs/2001.00309)] [[`BibTeX`](#citing-blendmask)]

This project contains training BlendMask for instance segmentation and panoptic segmentation on COCO and configs for segmenting persons on PIC.

## Quick Start

### Demo

```
wget -O blendmask_r101_dcni3_5x.pth https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download
python demo/demo.py \
    --config-file configs/BlendMask/R_101_dcni3_5x.yaml \
    --input datasets/coco/val2017/000000005992.jpg \
    --confidence-threshold 0.35 \
    --opts MODEL.WEIGHTS blendmask_r101_dcni3_5x.pth
```

### Training and evaluation

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),

Then follow [these steps](https://github.com/aim-uofa/AdelaiDet/tree/master/datasets#expected-dataset-structure-for-adelaidet-instance-detection) to generate blendmask format annotations for instance segmentation.

then run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 4 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x
```
To evaluate the model after training, run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --eval-only \
    --num-gpus 4 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x \
    MODEL.WEIGHTS training_dir/blendmask_R_50_1x/model_final.pth
```

## Models
### COCO Instance Segmentation Baselines

Model | Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:--:|
Mask R-CNN | [R_50_1x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 13 FPS | 38.6 | 35.2 |
BlendMask | [R_50_1x](configs/BlendMask/R_50_1x.yaml) | 14 FPS | 39.9 | 35.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/zoxXPnr6Hw3OJgK/download)
Mask R-CNN | [R_50_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) | 13 FPS | 41.0 | 37.2 | 
BlendMask | [R_50_3x](configs/BlendMask/R_50_3x.yaml) | 14 FPS | 42.7 | 37.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/ZnaInHFEKst6mvg/download)
Mask R-CNN | [R_101_3x](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) | 10 FPS | 42.9 | 38.6 |
BlendMask | [R_101_3x](configs/BlendMask/R_101_3x.yaml) | 11 FPS | 44.8 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/e4fXrliAcMtyEBy/download)
BlendMask | [R_101_dcni3_5x](configs/BlendMask/R_101_dcni3_5x.yaml) | 10 FPS | 46.8 | 41.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download)

### BlendMask Real-time Models

Model | Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:
Mask R-CNN | [550_R_50_3x](configs/RCNN/550_R_50_FPN_3x.yaml) | 16 FPS | 39.1 | 35.3 |
BlendMask | [550_R_50_3x](configs/BlendMask/550_R_50_3x.yaml) | 28 FPS | 38.7 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/R3Qintf7N8UCiIt/download)
BlendMask | [RT_R_50_4x](configs/BlendMask/RT_R_50_4x.yaml) | 30 FPS | 40.1 | 34.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/fmmciLkyaOoY1Tc/download)
BlendMask | [DLA_34_4x](configs/BlendMask/DLA_34_syncbn_4x.yaml) | 32 FPS | 40.9 | 35.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/Lx94rWNnZ8TRd2Y/download)

# Citing BlendMask
If you use BlendMask in your research or wish to refer to the baseline results, please use the following BibTeX entries.
```BibTeX
@inproceedings{chen2020blendmask,
  title     =  {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}
```