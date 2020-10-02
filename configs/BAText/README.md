# ABCNet (CVPR'20 oral)
[ABCNet](https://arxiv.org/abs/2002.10200) is an efficient end-to-end scene text spotting framework over 10x faster than previous SoTA. 

## Models
### CTW1500 results with ABCNet. 

Name | inf. time | e2e-hmean | det-hmean | download
--- |:---:|:---:|:---:|:---:
paper reported||45.2||
[attn_R_50](configs/BAText/CTW1500/attn_R_50.yaml) | 2080ti 8.7 FPS | 53.2 | 84.4 | [model](https://universityofadelaide.box.com/shared/static/okeo5pvul5v5rxqh4yg8pcf805tzj2no.pth)

### Total Text results with ABCNet. 

Name | inf. time | e2e-hmean | det-hmean | download
---  |:---------:|:---------:|:---------:|:---:
paper reported|V100 17.9 FPS|64.2||
[tt_attn_R_50](configs/BAText/TotalText/attn_R_50.yaml) | 2080ti 11.3 FPS | 67.1 | 86.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/tYsnegjTs13MwwK/download)
[pretrain_attn_R_50](configs/BAText/Pretrain/attn_R_50.yaml) | 2080ti 11.3 FPS | 58.1 | 80.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/dEzxhTlEumICiq0/download)

## Quick Start 

### Inference with our trained Models

1. Select the model and config file above, for example, `configs/BAText/CTW1500/attn_R_50.yaml`.
2. Run the demo with

```
wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/okeo5pvul5v5rxqh4yg8pcf805tzj2no.pth
python demo/demo.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --input datasets/CTW1500/ctwtest_text_image/ \
    --opts MODEL.WEIGHTS ctw1500_attn_R_50.pth
```
or
```
wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/tYsnegjTs13MwwK/download
python demo/demo.py \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --input datasets/totaltext/test_images/ \
    --opts MODEL.WEIGHTS tt_attn_R_50.pth
```
### Train Your Own Models

To train a model with "train_net.py", first setup the corresponding datasets following
[datasets/README.md](../../datasets/README.md). 

You can also prepare your custom dataset following the [example scripts](https://universityofadelaide.box.com/s/phqfzpvhe0obmkvn17akn9qw47u1m44i).

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

### Evaluate on Trained Model 
Download test GT [here](../../datasets/README.md) so that the directory has the following structure:

```
datasets
|_ evaluation
|  |_ gt_totaltext.zip
|  |_ gt_ctw1500.zip
```

Producing both e2e and detection results on CTW1500:
```
wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/okeo5pvul5v5rxqh4yg8pcf805tzj2no.pth
python tools/train_net.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --eval-only \
    MODEL.WEIGHTS ctw1500_attn_R_50.pth
```
or Totaltext:
```
wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/tYsnegjTs13MwwK/download
python tools/train_net.py \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --eval-only \
    MODEL.WEIGHTS tt_attn_R_50.pth
```

You can also evalute the json result file offline following the [evaluation_example_scripts](https://universityofadelaide.box.com/shared/static/e3yha5080jzvjuyfeayprnkbu265t3hr.zip), including an example of how to evaluate on a custom dataset. If you want to measure the ***inference time***, please change --num-gpus to 1.

# Cite

```BibTeX
@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

```

