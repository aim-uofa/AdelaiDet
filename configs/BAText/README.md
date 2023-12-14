# ABCNetv1 & ABCNetv2
[ABCNetv1](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.html) is an efficient end-to-end scene text spotting framework over 10x faster than previous state of the art. It's published in IEEE Conf. Comp Vis Pattern Recogn.'2020 as an oral paper. [ABCNetv2](https://ieeexplore.ieee.org/document/9525302) is published in TPAMI. 

## Models
### Experimental resutls on CTW1500: 

Name | inf. time | e2e-None-hmean | e2e-Full-hmean | det-hmean | download
--- |:---:|:---:|:---:|:---:|:---:
[v1-CTW1500-finetune](CTW1500/attn_R_50.yaml) |  8.7 FPS | 53.2 | 74.7 | 84.4 | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/ctw1500_attn_R_50.pth)
[v2-CTW1500-finetune](CTW1500/v2_attn_R_50.yaml) |  7.2 FPS | 57.7 | 75.8 | 85.0 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/model_v2_ctw1500.pth)


### Experimental resutls on TotalText:

Config | inf. time | e2e-None-hmean | e2e-Full-hmean | det-hmean | download
---  |:---------:|:---------:|:---------:|:---------:|:---:
[v1-pretrain](Pretrain/attn_R_50.yaml) |  11.3 FPS | 58.1 | 75.5 | 80.0 | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/pretrain_attn_R_50.pth)
[v1-totaltext-finetune](TotalText/attn_R_50.yaml) |  11.3 FPS | 67.1 | 81.1 | 86.0 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/tt_e2e_attn_R_50.pth)
[v2-pretrain](Pretrain/v2_attn_R_50.yaml) |  7.8 FPS | 63.5 | 78.4 | 83.7 | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/model_v2_pretrain.pth)
[v2-totaltext-finetune](TotalText/v2_attn_R_50.yaml) |  7.7 FPS | 71.8 | 83.4 | 87.2 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/model_v2_totaltext.pth)

### Experimental resutls on [ICDAR2015](https://rrc.cvc.uab.es/?ch=4):

Name | e2e-None | e2e-Generic | e2e-Weak | e2e-Strong | det-hmean | download
--- |:---:|:---:|:---:|:---:|:---:|:---:
[v1-icdar2015-pretrain](Pretrain/v1_ic15_attn_R_50.yaml) | 38.0 | 50.8 | 59.0 | 65.8 | 83.2 | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/v1_ic15_pretrained.pth)
[v1-icdar2015-finetune](ICDAR2015/v1_attn_R_50.yaml) | 57.1 | 66.8 | 74.1 | 79.2 | 86.8 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/v1_ic15_finetuned.pth)
[v2-icdar2015-pretrain](Pretrain/v2_ic15_attn_R_50.yaml) | 59.5 | 69.0 | 75.8 | 80.8 | 86.2 | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/ic15_pretrained.pth)
[v2-icdar2015-finetune](ICDAR2015/v2_attn_R_50.yaml) | 66.3 | 73.2 | 78.8 | 83.7 | 88.2 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/ic15_finetuned.pth)

### Experimental resutls on [ReCTS](https://rrc.cvc.uab.es/?ch=12):

Name | inf. time | det-recall | det-precision | det-hmean | 1 - NED | download
--- |:---:|:---:|:---:|:---:|:---:|:---:
[v2-Chinese-pretrained](Pretrain/v2_chn_attn_R_50.yaml) |  -| - | - | - | - | [pretrained-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/model_v2_chn_pretrain.pth)
[v2-ReCTS-finetune](ReCTS/v2_chn_attn_R_50.yaml) |  8 FPS | 87.9 | 92.9 | 90.33 | 63.9 | [finetuned-model](https://huggingface.co/ZjuCv/AdelaiDet/blob/main/model_v2_rects.pth)


### Experimental resutls on [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_%28MSRA-TD500%29):

Name | det-recall | det-precision | det-hmean | download
--- |:---:|:---:|:---:|:---:
[v2-TD500-finetune](https://github.com/aim-uofa/AdelaiDet/issues/537) | 81.9 | 89.0 | 85.3 | [finetuned-model](https://github.com/aim-uofa/AdelaiDet/issues/537)

* Note the pretrained model for TD500 is the Chinese pretrained used for ReCTS. As MSRA-TD is a det. only dataset, a small amount of [modification](https://github.com/aim-uofa/AdelaiDet/issues/537) is needed.

## Quick Start (ABCNetv1)

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
or
```
# Download v1_ic15_finetuned.pth above
python demo/demo.py \
    --config-file configs/BAText/ICDAR2015/v1_attn_R_50.yaml \
    --input datasets/icdar2015/test_images \
    --opts MODEL.WEIGHTS v1_ic15_finetuned.pth
```
### Train Your Own Models

To train a model with "train_net.py", first setup the corresponding datasets following
[datasets/README.md](../../datasets/README.md) or using the following script:

```
cd datasets/
wget https://drive.google.com/file/d/1we4iwZNA80q-yRoEKqB66SuTa1tPbhZu/view?usp=sharing -O totaltext.zip
unzip totaltext.zip
rm totaltext.zip
wget https://drive.google.com/file/d/1ntlnlnQHZisDoS_bgDvrcrYFomw9iTZ0/view?usp=sharing -O CTW1500.zip
unzip CTW1500.zip 
rm CTW1500.zip
wget https://drive.google.com/file/d/1J94245rU-s7KTecNQRD3KXG04ICZhL9z/view?usp=sharing -O icdar2015.zip
unzip icdar2015.zip 
rm icdar2015.zip
mkdir evaluation
cd evaluation
wget -O gt_ctw1500.zip https://cloudstor.aarnet.edu.au/plus/s/xU3yeM3GnidiSTr/download
wget -O gt_totaltext.zip https://cloudstor.aarnet.edu.au/plus/s/SFHvin8BLUM4cNd/download
wget -O gt_icdar2015.zip https://drive.google.com/file/d/1wrq_-qIyb_8dhYVlDzLZTTajQzbic82Z/view?usp=sharing
```

* Note (synthetic and mlt2017 datasets need to be downloaded through [datasets/README.md](../../datasets/README.md).)

You can also prepare your custom dataset following the [example scripts](https://universityofadelaide.box.com/s/phqfzpvhe0obmkvn17akn9qw47u1m44i).

Pretrainining with synthetic data (For Totaltext and CTW1500):

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/attn_R_50
```

Pretrainining with synthetic data (For ICDAR2015):

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/v1_ic15_attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/v1_ic15_attn_R_50
```


Finetuning on TotalText:

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

Finetuning on ICDAR2015:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/ICDAR2015/v1_attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/v1_ic15_attn_R_50/model_final.pth
```

### Evaluate on Trained Model 
Download test GT [here](../../datasets/README.md) so that the directory has the following structure:

```
datasets
|_ evaluation
|  |_ gt_totaltext.zip
|  |_ gt_ctw1500.zip
|  |_ gt_icdar2015.zip
```

Producing both (w/wo lexion) e2e and detection results on CTW1500:
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
or ICDAR2015:
```
# Download v1_ic15_finetuned.pth above
# MODEL.BATEXT.EVAL_TYPE: 3: Strong, 2: Weak, 1: Generic
python tools/train_net.py \
    --config-file configs/BAText/ICDAR2015/v1_attn_R_50.yaml \
    --num-gpus 4 \
    --eval-only \
    MODEL.WEIGHTS v1_ic15_finetuned.pth \
    MODEL.BATEXT.EVAL_TYPE 3     
```

You can also evalute the json result file offline following the [evaluation_example_scripts](https://universityofadelaide.box.com/shared/static/e3yha5080jzvjuyfeayprnkbu265t3hr.zip), including an example of how to evaluate on a custom dataset. If you want to measure the ***inference time***, please change --num-gpus to 1.

### Standalone BezierAlign Warping 
If you are insteresting in warping a curved instance into a rectangular format independantly, please refer to the example script [here](https://github.com/Yuliang-Liu/bezier_curve_text_spotting#bezieralign-example).

## Quick Start (ABCNetv2)
The datasets and the basic training details (learning rate, iterations, etc.) used for ABCNetv2 are exactly the same as ABCNet v1. Please following above to prepare the training and evaluation data. If you are interesting in text spotting quantization, please refer to the [patch](https://github.com/aim-uofa/model-quantization/blob/master/doc/detectron2.md#text-spotting).

### Demo
* For CTW1500
```
# Download model_v2_ctw1500.pth above
python demo/demo.py \
    --config-file configs/BAText/CTW1500/v2_attn_R_50.yaml \
    --input datasets/CTW1500/ctwtest_text_image/ \
    --opts MODEL.WEIGHTS model_v2_ctw1500.pth
```
* For TotalText
```
# Download model_v2_totaltext.pth above
python demo/demo.py \
    --config-file configs/BAText/TotalText/v2_attn_R_50.yaml \
    --input datasets/totaltext/test_images/ \
    --opts MODEL.WEIGHTS model_v2_totaltext.pth
```
* For ICDAR2015
```
# Download ic15_finetuned.pth above
python demo/demo.py \
    --config-file configs/BAText/ICDAR2015/v2_attn_R_50.yaml \
    --input datasets/icdar2015/test_images/ \
    --opts MODEL.WEIGHTS ic15_finetuned.pth
```
* For ReCTS (Chinese)
```
# Download model_v2_rects.pth above
wget https://drive.google.com/file/d/1dcR__ZgV_JOfpp8Vde4FR3bSR-QnrHVo/view?usp=sharing -O simsun.ttc
wget https://drive.google.com/file/d/1wqkX2VAy48yte19q1Yn5IVjdMVpLzYVo/view?usp=sharing -O chn_cls_list
python demo/demo.py \
    --config-file configs/BAText/ReCTS/v2_chn_attn_R_50.yaml \
    --input datasets/ReCTS/ReCTS_test_images/ \
    --opts MODEL.WEIGHTS model_v2_rects.pth
```

### Train
Training ABCNetv2 using 4 V100.
* Pretrainining with synthetic data (for TotalText and CTW1500):
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/v2_attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/v2_attn_R_50
```
* Pretrainining with synthetic data (for ICDAR2015):
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/v2_ic15_attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/v2_ic15_attn_R_50
```
* Pretrainining with synthetic data (for ReCTS):
```
wget https://drive.google.com/file/d/1wqkX2VAy48yte19q1Yn5IVjdMVpLzYVo/view?usp=sharing -O chn_cls_list
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/Pretrain/v2_chn_attn_R_50.yaml \
    --num-gpus 4 \
    OUTPUT_DIR text_pretraining/v2_chn_attn_R_50
```
* Finetuning on TotalText:
```
# Download model_v2_pretrain.pth above or using your own pretrained model
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/TotalText/v2_attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/v2_attn_R_50/model_final.pth
```
* Finetuning on CTW1500:
```
# Download model_v2_pretrain.pth above or using your own pretrained model
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/CTW1500/v2_attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS text_pretraining/v2_attn_R_50/model_final.pth
```
* Finetuning on ICDAR2015:
```
# Download ic15_pretrained.pth above or using your own pretrained model
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/ICDAR2015/v2_attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS ic15_pretrained.pth
```
* Finetuning on ReCTS:
```
# Download model_v2_chn_pretrain.pth or using your own pretrained model
wget https://drive.google.com/file/d/1XOtlUz9lxh2HV5Gmu3alb5WKZafFn-0_/view?usp=sharing -O model_v2_chn_pretrain.pth
wget https://drive.google.com/file/d/1wqkX2VAy48yte19q1Yn5IVjdMVpLzYVo/view?usp=sharing -O chn_cls_list
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BAText/ReCTS/v2_chn_attn_R_50.yaml \
    --num-gpus 4 \
    MODEL.WEIGHTS model_v2_chn_pretrain.pth
```
### Evaluation 
* Evaluate on CTW1500:
```
# Download model_v2_ctw1500.pth above
python tools/train_net.py \
    --config-file configs/BAText/CTW1500/v2_attn_R_50.yaml \
    --eval-only \
    MODEL.WEIGHTS model_v2_ctw1500.pth
```
* Evaluate on Totaltext:
```
# Download model_v2_totaltext.pth above
python tools/train_net.py \
    --config-file configs/BAText/TotalText/v2_attn_R_50.yaml \
    --eval-only \
    MODEL.WEIGHTS model_v2_totaltext.pth
```
* Evaluate on ICDAR2015:
```
# Download ic15_finetuned.pth above 
# MODEL.BATEXT.EVAL_TYPE: 3: Strong, 2: Weak, 1: Generic
python tools/train_net.py \
    --config-file configs/BAText/ICDAR2015/v2_attn_R_50.yaml \
    --num-gpus 4 \
    --eval-only \
    MODEL.WEIGHTS ic15_finetuned.pth \
    MODEL.BATEXT.EVAL_TYPE 3 
```
* Evaluate on ReCTS:

ReCTS does not provide annotations for the test set, you may need to submit the results using the predicted json file in the [official website](https://rrc.cvc.uab.es/?ch=12).


# BibTeX

```BibTeX
@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}
@ARTICLE{9525302,
  author={Liu, Yuliang and Shen, Chunhua and Jin, Lianwen and He, Tong and Chen, Peng and Liu, Chongyu and Chen, Hao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3107437}}
```

