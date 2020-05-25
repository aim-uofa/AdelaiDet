# ABCNet
[ABCNet](https://arxiv.org/abs/2002.10200) is an efficient end-to-end scene text spotting framework. Note that the implementation details have slight different with original paper (less data, different recognition decoder, etc). 

## Models
### CTW1500 results with ABCNet. 

Name | inf. time | e2e-hmean | det-hmean | download
--- |:---:|:---:|:---:|:---:
paper reported||45.2||
[attn_R_50](configs/BAText/CTW1500/attn_R_50.yaml) | 2080ti 8.7 FPS | 53.2 | 84.4 | [model](https://universityofadelaide.box.com/shared/static/1bqpg9hijtn2rcooqjpffateguh9eeme.pth)

### Total Text results with ABCNet. 

Name | inf. time | e2e-hmean | det-hmean | download
---  |:---------:|:---------:|:---------:|:---:
paper reported|V100 17.9 FPS|64.2||
[tt_attn_R_50](configs/BAText/TotalText/attn_R_50.yaml) | 2080ti 11.3 FPS | 67.1 | 86.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download)
[pretrain_attn_R_50](configs/BAText/Pretrain/attn_R_50.yaml) | 2080ti 11.3 FPS | 58.1 | 80.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/UenknKbsWAuBUcz/download)

## Quick Start 

### Inference with our trained Models

1. Select the model and config file above, for example, `configs/BAText/CTW1500/attn_R_50.yaml`.
2. Run the demo with

```
wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/1bqpg9hijtn2rcooqjpffateguh9eeme.pth
python demo/demo.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --input datasets/CTW1500/ctwtest_text_image/ \
    --opts MODEL.WEIGHTS ctw1500_attn_R_50.pth
```
or
```
wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download
python demo/demo.py \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --input datasets/totaltext/test_images/ \
    --opts MODEL.WEIGHTS tt_attn_R_50.pth
```
### Train Your Own Models

To train a model with "train_net.py", first setup the corresponding datasets following
[datasets/README.md](../../datasets/README.md). 

You can also prepare your custom dataset using the example scripts:
- Generate Bezier-curve control points from polygons. [[link]](https://drive.google.com/file/d/1bFmdXCCsW0bj0qFgQl1MJarlWkwPSv_U/view)
- Generate Bezier-curve json file. [[link]](https://universityofadelaide.box.com/s/ytfhetwat4fqnq4ptfprxu93wp4zb4as)


Pretrainining with synthetic data:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/attn_R_50
```

Finetuning on Total Text:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/attn_R_50/model_final.pth
```

Finetuning on CTW1500:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/attn_R_50/model_final.pth
```

For evaluation only (we will produce both e2e and detection results):
```
python tools/train_net.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --eval-only \
    MODEL.WEIGHTS your_model.pth
```

# Cite

```BibTeX
@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

```


