# Conditional Convolutions for Instance Segmentation (Oral)

    Conditional Convolutions for Instance Segmentation;
    Zhi Tian, Chunhua Shen and Hao Chen;
    In: Proc. European Conference on Computer Vision (ECCV), 2020.
    arXiv preprint arXiv:2003.05664

[[`Paper`](https://arxiv.org/abs/2003.05664)] [[`BibTeX`](#citing-condinst)]


# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

For demo, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/M8nNxSR5iNP4qyO/download -O CondInst_MS_R_101_3x_sem.pth
python demo/demo.py \
    --config-file configs/CondInst/MS_R_101_3x_sem.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS CondInst_MS_R_101_3x_sem.pth
```

For training on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/CondInst/MS_R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/CondInst_MS_R_50_1x
```

For evaluation on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/CondInst/MS_R_50_1x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/CondInst_MS_R_50_1x \
    MODEL.WEIGHTS training_dir/CondInst_MS_R_50_1x/model_final.pth
```


## Models
### COCO Instance Segmentation Baselines with [CondInst](https://arxiv.org/abs/2003.05664)

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[CondInst_MS_R_50_1x](MS_R_50_1x.yaml) | 14 FPS | 39.7 | 35.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/Trx1r4tLJja7sLT/download)
[CondInst_MS_R_50_3x](MS_R_50_3x.yaml) | 14 FPS | 41.9 | 37.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/T3OGVBiaSVLvo5E/download)
[CondInst_MS_R_101_3x](MS_R_101_3x.yaml) | 11 FPS | 43.3 | 38.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/vWLiYm8OnrTSUD2/download)

With an auxiliary semantic segmentation task (set `MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = True` to enable it):

Name | inf. time | box AP | mask AP | mask AP (test-dev) | download
--- |:---:|:---:|:---:|:---:|:---:
[CondInst_MS_R_50_3x_sem](MS_R_50_3x_sem.yaml) | 14 FPS | 42.6 | 38.2 | 38.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/75Ag8VvC6WedVNh/download)
[CondInst_MS_R_101_3x_sem](MS_R_101_3x_sem.yaml) | 11 FPS | 44.6 | 39.8 | 40.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/M8nNxSR5iNP4qyO/download)

With BiFPN:

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[CondInst_MS_R_50_BiFPN_1x](MS_R_50_BiFPN_1x.yaml) | 13 FPS | 42.5 | 37.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/RyCG82WhTop99j2/download)
[CondInst_MS_R_50_BiFPN_3x](MS_R_50_BiFPN_3x.yaml) | 13 FPS | 44.3 | 38.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/W9ZCcxJF0P5NhJQ/download)
[CondInst_MS_R_50_BiFPN_3x_sem](MS_R_50_BiFPN_3x_sem.yaml) | 13 FPS | 44.7 | 39.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/9cAHjZtdaAGnb2Q/download)
[CondInst_MS_R_101_BiFPN_3x](MS_R_101_BiFPN_3x.yaml) | 10 FPS | 45.3 | 39.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/HyB0O0D7hfpUC2n/download)
[CondInst_MS_R_101_BiFPN_3x_sem](MS_R_101_BiFPN_3x_sem.yaml) | 10 FPS | 45.7 | 40.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/2p1ashxl54Su8vv/download)


*Disclaimer:*

- All models are trained with multi-scale data augmentation. Inference time is measured on a single NVIDIA 1080Ti with batch size 1.
- The final mask's resolution is 1/4 of the input image (i.e., `MODEL.CONDINST.MASK_OUT_STRIDE = 4`, which is enough on MS-COCO and different from our original paper. In the paper, we used `MODEL.CONDINST.MASK_OUT_STRIDE = 2`. If you want high-resolution mask results, please reduce it.
- This is a reimplementation. Thus, the numbers are slightly different from our original paper (within 0.1% in mask AP).


# Citing CondInst
If you use CondInst in your research or wish to refer to the baseline results, please use the following BibTeX entries.
```BibTeX
@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```
