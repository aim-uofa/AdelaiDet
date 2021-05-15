#  Dense Contrastive Learning for Self-Supervised Visual Pre-Training

Here we provide instructions and results for applying DenseCL pre-trained models to AdelaiDet. Please refer to [https://git.io/DenseCL
](https://git.io/DenseCL
) for the pre-training code.

> [**Dense Contrastive Learning for Self-Supervised Visual Pre-Training**](https://arxiv.org/abs/2011.09157),  
> Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, Lei Li   
> In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2021, **Oral**  
> *arXiv preprint ([arXiv 2011.09157](https://arxiv.org/abs/2011.09157))*   


# Installation 
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).


# DenseCL Pre-trained Models
pre-train method | pre-train dataset | backbone | #epoch | Link
--- |:---:|:---:|:---:|:---:
DenseCL | COCO | ResNet-50 | 800 | [download](https://cloudstor.aarnet.edu.au/plus/s/W5oDyYB218xz625/download)
DenseCL | COCO | ResNet-50 | 1600 |  [download](https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download)
DenseCL | ImageNet | ResNet-50 | 200 |  [download](https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download)
DenseCL | ImageNet | ResNet-101 | 200 | [download](https://cloudstor.aarnet.edu.au/plus/s/4sugyvuBOiMXXnC/download)


# Usage

## Download the pre-trained model
```
PRETRAIN_DIR=./
wget https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download -O ${PRETRAIN_DIR}/densecl_r50_imagenet_200ep.pkl
```

## Convert it to detectron2's format
Use [convert-pretrain-to-detectron2.py](https://github.com/WXinlong/DenseCL/blob/main/benchmarks/detection/convert-pretrain-to-detectron2.py) to convert the pre-trained backbone weights:
```
WEIGHT_FILE=${PRETRAIN_DIR}/densecl_r50_imagenet_200ep.pth
OUTPUT_FILE=${PRETRAIN_DIR}/densecl_r50_imagenet_200ep.pkl
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE}
```

## Train the downstream models

For training a SOLOv2, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/DenseCL/SOLOv2_R50_1x_DenseCL.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_1x_DenseCL \
    MODEL.WEIGHTS ${PRETRAIN_DIR}/densecl_r50_imagenet_200ep.pkl
```

For training a FCOS, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/DenseCL/FCOS_R50_1x_DenseCL.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/FCOS_R50_1x_DenseCL \
    MODEL.WEIGHTS ${PRETRAIN_DIR}/densecl_r50_imagenet_200ep.pkl
```


# Performance
## SOLOv2 on COCO Instance Segmentation

pre-train method | pre-train dataset  |  mask AP | 
--- |:---:|:---:|
Supervised  | ImageNet | 35.2  
MoCo-v2  | ImageNet | 35.2
DenseCL |  ImageNet | 35.7 (+0.5)

## FCOS on COCO Object Detection

pre-train method | pre-train dataset  |  box AP | 
--- |:---:|:---:|
Supervised   | ImageNet | 39.9
MoCo-v2  | ImageNet | 40.3
DenseCL |  ImageNet | 40.9 (+1.0)



# Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```BibTeX
@inproceedings{wang2020densecl,
  title     =   {Dense Contrastive Learning for Self-Supervised Visual Pre-Training},
  author    =   {Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  booktitle =   {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2021}
}
```
