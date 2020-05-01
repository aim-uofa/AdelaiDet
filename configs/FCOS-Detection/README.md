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

Name | inf. time | box AP | download
--- |:---:|:---:|:---:
[FCOS_R_50_1x](R_50_1x.yaml) | 16 FPS | 38.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download)
[FCOS_MS_R_50_2x](MS_R_50_2x.yaml) | 16 FPS | 41.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/reA6HVaGX47yKGV/download)
[FCOS_MS_R_101_2x](MS_R_101_2x.yaml) | 12 FPS | 43.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/M3UOT6JcyHy2QW1/download)
[FCOS_MS_X_101_32x8d_2x](MS_X_101_32x8d_2x.yaml) | 6.6 FPS | 43.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/R7H00WeWKZG45pP/download)
[FCOS_MS_X_101_64x4d_2x](MS_X_101_64x4d_2x.yaml) | 6.1 FPS | 44.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/XOLUCzqKYckNII7/download)
[FCOS_MS_X_101_32x8d_dcnv2_2x](MS_X_101_32x8d_2x_dcnv2.yaml) | 4.6 FPS | 46.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/TDsnYK8OXDTrafF/download)

*Except for FCOS_R_50_1x, all other models are trained with multi-scale data augmentation.*

### FCOS Real-time Models

Name | inf. time | box AP | download
--- |:---:|:---:|:---:
[FCOS_RT_MS_DLA_34_4x_shtw](FCOS_RT/MS_DLA_34_4x_syncbn_shared_towers.yaml) | 52 FPS | 39.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/4vc3XwQezyhNvnB/download)
[FCOS_RT_MS_DLA_34_4x](FCOS_RT/MS_DLA_34_4x_syncbn.yaml) | 46 FPS | 40.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/zNPNyTkizaOOsUQ/download)
[FCOS_RT_MS_R_50_4x](FCOS_RT/MS_R_50_4x_syncbn.yaml) | 38 FPS | 40.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/TlnlXUr6lNNSyoZ/download)

If you prefer BN in FCOS heads, please try the following models.

Name | inf. time | box AP | download
--- |:---:|:---:|:---:
[FCOS_RT_MS_DLA_34_4x_shtw_bn](FCOS_RT/MS_DLA_34_4x_syncbn_shared_towers_bn_head.yaml) | 52 FPS | 38.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/rdmHHSs4oCg7l7U/download)
[FCOS_RT_MS_DLA_34_4x_bn](FCOS_RT/MS_DLA_34_4x_syncbn_bn_head.yaml) | 48 FPS | 39.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/T5httPVo1VndbD4/download)
[FCOS_RT_MS_R_50_4x_bn](FCOS_RT/MS_R_50_4x_syncbn_bn_head.yaml) | 40 FPS | 39.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/dHNUNs0YxVhZAmg/download)

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
