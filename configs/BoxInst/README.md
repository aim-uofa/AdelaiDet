# BoxInst: High-Performance Instance Segmentation with Box Annotations

    BoxInst: High-Performance Instance Segmentation with Box Annotations;
    Zhi Tian, Chunhua Shen, Xinlong Wang and Hao Chen;
    In: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2021.
    arXiv preprint arXiv:2012.02310

[[`Paper`](https://arxiv.org/abs/2012.02310)] [[`BibTeX`](#citing-boxinst)] [[`Video Demo`](https://www.youtube.com/watch?v=NuF8NAYf5L8)]


# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

For demo, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/Aabn3BEuq4HKiNK/download -O BoxInst_MS_R_50_3x.pth
python demo/demo.py \
    --config-file configs/BoxInst/MS_R_50_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS BoxInst_MS_R_50_3x.pth
```

For training on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BoxInst/MS_R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/BoxInst_MS_R_50_1x
```

For evaluation on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BoxInst/MS_R_50_1x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/BoxInst_MS_R_50_1x \
    MODEL.WEIGHTS training_dir/BoxInst_MS_R_50_1x/model_final.pth
```


## Models
### COCO Instance Segmentation Baselines with [BoxInst](https://arxiv.org/abs/2012.02310)

Only **box annotations** are used during training.

Name | inf. time | box AP | mask AP | mask AP (test-dev)| download
--- |:---:|:---:|:---:|:---:|:---:
[BoxInst_MS_R_50_1x](MS_R_50_1x.yaml) | 14 FPS | 39.4 | 30.7 | - | [model](https://cloudstor.aarnet.edu.au/plus/s/odj8VwqgRT8TMsR/download)
[BoxInst_MS_R_50_3x](MS_R_50_3x.yaml) | 14 FPS | 41.5 | 31.8 | 32.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/Aabn3BEuq4HKiNK/download)
[BoxInst_MS_R_101_1x](MS_R_101_1x.yaml) | 11 FPS | 41.4 | 32.2 | 32.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/FiPXJJ1GdJtdO0w/download)
[BoxInst_MS_R_101_3x](MS_R_101_3x.yaml) | 11 FPS | 43.3 | 33.0 | 33.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/8CcXfqKpPFc4bJ4/download)
[BoxInst_MS_R_101_BiFPN_3x](MS_R_101_BiFPN_3x.yaml) | 10 FPS | 45.4 | 34.1 | 33.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/qGGrOlYgTqIur7B/download)
[BoxInst_MS_R_101_BiFPN_dcni3_3x](MS_R_101_BiFPN_dcni3_3x.yaml) | 8 FPS | 46.4 | 34.8 | 35.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/e8hivzBFhadkEfc/download)

Disclaimer:
- All models are trained with multi-scale data augmentation. Inference time is measured on a single NVIDIA 1080Ti with batch size 1.
- This is a reimplementation. Thus, the numbers might be slightly different from the ones reported in our original paper.


# Citing BoxInst
If you use BoxInst in your research or wish to refer to the baseline results, please use the following BibTeX entries.
```BibTeX
@inproceedings{tian2020boxinst,
  title     =  {{BoxInst}: High-Performance Instance Segmentation with Box Annotations},
  author    =  {Tian, Zhi and Shen, Chunhua and Wang, Xinlong and Chen, Hao},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2021}
}
```
