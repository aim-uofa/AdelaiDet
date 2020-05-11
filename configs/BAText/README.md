# ABCNet
[ABCNet](https://arxiv.org/abs/2002.10200) is an efficient end-to-end scene text spotting framework. Note that the implementation details have slight different with original paper (less data, different recognition decoder, etc).

## Models
### CTW15000 results with ABCNet.

Name | inf. time | e2e-hmean | det-hmean | download
--- |:---:|:---:|:---:|:---:
[attn_R_50](configs/BAText/CTW1500/attn_R_50.yaml) | 8.7 FPS | 45.2 | 81.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/AexPQx2gB7CrCnQ/download)

* E2E result is achieved with threshold 0.6 and Det with 0.5.

### Total Text results with ABCNet.

Name | inf. time | e2e-hmean | det-hmean | download
---  |:---------:|:---------:|:---------:|:---:
[attn_R_50](configs/BAText/TotalText/attn_R_50.yaml) | 11 FPS | 63.0 | 82.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/nyyNRdP7VBYqfgl/download)

* E2E result is achieved with threshold 0.6 and Det with 0.55.


## Quick Start 

### Inference with our trained Models

1. Select the model and config file above, for example, `configs/BAText/CTW1500/attn_R_50.yaml`.
2. Run the demo with

```
wget -O ctw1500_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/AexPQx2gB7CrCnQ/download
python demo/demo.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --input input.jpg \
    --opts MODEL.WEIGHTS ctw1500_attn_R_50.pth
```

### Train Your Own Models

To train a model with "train_net.py", first setup the corresponding datasets following
[datasets/README.md](../../datasets/README.md).

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

Finetuning on CTW 1500:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/attn_R_50/model_final.pth
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


