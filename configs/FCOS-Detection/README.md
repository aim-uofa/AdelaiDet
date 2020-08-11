# FCOS: Fully Convolutional One-Stage Object Detection

    FCOS: Fully Convolutional One-Stage Object Detection;
    Zhi Tian, Chunhua Shen, Hao Chen, and Tong He;
    In: Proc. Int. Conf. Computer Vision (ICCV), 2019.
    arXiv preprint arXiv:1904.01355 

[[`Paper`](https://arxiv.org/abs/1904.01355)] [[`BibTeX`](#citing-fcos)]

# Installation & Quick Start
No special setup needed. The [default instruction](../../README.md#Installation) is fine.

## Models
### COCO Object Detecton Baselines with [FCOS](https://arxiv.org/abs/1904.01355)

Name | inf. time | box AP | box AP (test-dev) | download
--- |:---:|:---:|:---:|:---:
[FCOS_R_50_1x](R_50_1x.yaml) | 16 FPS | 38.7 | [38.8](https://gist.github.com/tianzhi0549/1c8d115efaf1e49a4f390cce63ca69ca) | [model](https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download)
[FCOS_MS_R_50_2x](MS_R_50_2x.yaml) | 16 FPS | 41.0 | [41.4](https://gist.github.com/tianzhi0549/3ca076c2125891312dbf5ce932469e76) | [model](https://cloudstor.aarnet.edu.au/plus/s/reA6HVaGX47yKGV/download)
[FCOS_MS_R_101_2x](MS_R_101_2x.yaml) | 12 FPS | 43.1 | [43.2](https://gist.github.com/tianzhi0549/d97994a5b72980ba94de25737d2d40cb) | [model](https://cloudstor.aarnet.edu.au/plus/s/M3UOT6JcyHy2QW1/download)
[FCOS_MS_X_101_32x8d_2x](MS_X_101_32x8d_2x.yaml) | 6.6 FPS | 43.9 | [44.1](https://gist.github.com/tianzhi0549/3135d6e0fad24b07cc685fef660c5363) | [model](https://cloudstor.aarnet.edu.au/plus/s/R7H00WeWKZG45pP/download)
[FCOS_MS_X_101_64x4d_2x](MS_X_101_64x4d_2x.yaml) | 6.1 FPS | 44.7 | [44.8](https://gist.github.com/tianzhi0549/b68f6500ec24e6b263c12c345a7b5c7b) | [model](https://cloudstor.aarnet.edu.au/plus/s/XOLUCzqKYckNII7/download)
[FCOS_MS_X_101_32x8d_dcnv2_2x](MS_X_101_32x8d_2x_dcnv2.yaml) | 4.6 FPS | 46.6 | [46.6](https://gist.github.com/tianzhi0549/316e8feaa17bf0341e2effa485fb41c0) | [model](https://cloudstor.aarnet.edu.au/plus/s/TDsnYK8OXDTrafF/download)

*Except for FCOS_R_50_1x, all other models are trained with multi-scale data augmentation.*

### FCOS Real-time Models

Name | inf. time | box AP | box AP (test-dev) | download
--- |:---:|:---:|:---:|:---:
[FCOS_RT_MS_DLA_34_4x_shtw](FCOS_RT/MS_DLA_34_4x_syncbn_shared_towers.yaml) | 52 FPS | 39.1 | [39.2](https://gist.github.com/tianzhi0549/9f56ceaec77e2eb4170b6cd18da2856c) | [model](https://cloudstor.aarnet.edu.au/plus/s/4vc3XwQezyhNvnB/download)
[FCOS_RT_MS_DLA_34_4x](FCOS_RT/MS_DLA_34_4x_syncbn.yaml) | 46 FPS | 40.3 | [40.3](https://gist.github.com/tianzhi0549/338d8614beafe21b7af4dc5defc37d95) | [model](https://cloudstor.aarnet.edu.au/plus/s/zNPNyTkizaOOsUQ/download)
[FCOS_RT_MS_R_50_4x](FCOS_RT/MS_R_50_4x_syncbn.yaml) | 38 FPS | 40.2 | [40.2](https://gist.github.com/tianzhi0549/5c7892831d9c03d615214a66e3af19f4) | [model](https://cloudstor.aarnet.edu.au/plus/s/TlnlXUr6lNNSyoZ/download)

If you prefer BN in FCOS heads, please try the following models.

Name | inf. time | box AP | box AP (test-dev) | download
--- |:---:|:---:|:---:|:---:
[FCOS_RT_MS_DLA_34_4x_shtw_bn](FCOS_RT/MS_DLA_34_4x_syncbn_shared_towers_bn_head.yaml) | 52 FPS | 38.9 | [39.1](https://gist.github.com/tianzhi0549/d87298bb7beb7c926a355708d05e9a0c) | [model](https://cloudstor.aarnet.edu.au/plus/s/rdmHHSs4oCg7l7U/download)
[FCOS_RT_MS_DLA_34_4x_bn](FCOS_RT/MS_DLA_34_4x_syncbn_bn_head.yaml) | 48 FPS | 39.4 | [39.9](https://gist.github.com/tianzhi0549/6a7053943c96111134a81f3141d1b9b5) | [model](https://cloudstor.aarnet.edu.au/plus/s/T5httPVo1VndbD4/download)
[FCOS_RT_MS_R_50_4x_bn](FCOS_RT/MS_R_50_4x_syncbn_bn_head.yaml) | 40 FPS | 39.3 | [39.7](https://gist.github.com/tianzhi0549/35869c1d00688b4d60cc8f7e7d91c94d) | [model](https://cloudstor.aarnet.edu.au/plus/s/dHNUNs0YxVhZAmg/download)

*Inference time is measured on a NVIDIA 1080Ti with batch size 1. Real-time models use shorter side 512 for inference.*

# Citing FCOS
If you use FCOS in your research or wish to refer to the baseline results, please use the following BibTeX entries.
```BibTeX
@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}
```
