# FCPose: Fully Convolutional Multi-Person Pose Estimation with Dynamic Instance-Aware Convolutions



# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

For training on COCO, run:
```
python tools/train_net.py \
    --num-gpus 8 \
    --config-file configs/FCPose/R_50_3X.yaml \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    OUTPUT_DIR training_dir/R_50_3X
```

For evaluation on COCO, run:
```
python tools/train_net.py \
    --num-gpus 8 \
    --eval-only \
    --config-file configs/FCPose/R_50_3X.yaml \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    OUTPUT_DIR training_dir/R_50_3X \
    MODEL.WEIGHTS training_dir/R_50_3X/model_final.pth
```


## Models
### COCO Instance Segmentation Baselines with SOLOv2

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[FCPose_R50_3x](R_50_3X.yaml) | 45ms | 57.9  | 65.2  | [model](https://cloudstor.aarnet.edu.au/plus/s/KoM3QUu9FMbSG1Q/download)
[FCPose_R101_3x](R_101_3X.yaml) | 58ms | 58.7  | 67.0  | [model](https://cloudstor.aarnet.edu.au/plus/s/kNVlDpPNHcOsRFf/download)


*Disclaimer:*

- Inference time is measured on 8 V100 GPUs.
- This is a reimplementation. Thus, the numbers are slightly different from our original paper.
- This is a alpha version. We will update our implement later, including adding real-time version FCPose and fixing the issue of the loss being nan. if you found you loss being nan when training, please try again.


# Citations
Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.
```BibTeX
@inproceedings{mao2021fcpose,
  title={FCPose: Fully Convolutional Multi-Person Pose Estimation with Dynamic Instance-Aware Convolutions},
  author={Mao, Weian and Tian, Zhi and Wang, Xinlong and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9034--9043},
  year={2021}
}
```
