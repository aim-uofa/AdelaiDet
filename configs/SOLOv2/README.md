# SOLOv2: Dynamic and Fast Instance Segmentation


> [**SOLOv2: Dynamic and Fast Instance Segmentation**](https://arxiv.org/abs/2003.10152),            
> Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, Chunhua Shen     
> In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020  
> *arXiv preprint ([arXiv 2003.10152](https://arxiv.org/abs/2003.10152))*  



# Installation & Quick Start
First, follow the [default instruction](../../README.md#Installation) to install the project and [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) 
set up the datasets (e.g., MS-COCO).

For demo, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download -O SOLOv2_R50_3x.pth
python demo/demo.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth
```

For training on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x
```

For evaluation on COCO, run:
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/SOLOv2/R50_3x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x \
    MODEL.WEIGHTS training_dir/SOLOv2_R50_3x/model_final.pth
```


## Models
### COCO Instance Segmentation Baselines with SOLOv2

Name | inf. time | train. time | Mem | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:|:---:
[SOLOv2_R50_3x](R50_3x.yaml) | 47ms | ~25h(36 epochs) | 3.7GB  | -  | 37.6  | [model](https://cloudstor.aarnet.edu.au/plus/s/chF3VKQT4RDoEqC/download)
[SOLOv2_R101_3x](R101_3x.yaml) | 61ms | ~30h(36 epochs) | 4.7GB | -   | 39.0  | [model](https://cloudstor.aarnet.edu.au/plus/s/9w7b3sjaXvqYQEQ)


*Disclaimer:*

- All models are trained with multi-scale data augmentation. 
- Inference time is measured on a single V100 GPU. Training time is measured on 8 V100 GPUs.
- This is a reimplementation. Thus, the numbers are slightly different from our original paper (within 0.3% in mask AP).
- The implementation on mmdetection is available at [https://github.com/WXinlong/SOLO](https://github.com/WXinlong/SOLO).


# Citations
Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.
```BibTeX
@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}

```

```BibTeX
@inproceedings{wang2020solov2,
  title   =  {{SOLOv2}: Dynamic and Fast Instance Segmentation},
  author  =  {Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  booktitle =  {Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year    =  {2020}
}
```

```BibTeX
@article{wang2021solo,
  title   =  {{SOLO}: A Simple Framework for Instance Segmentation},
  author  =  {Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  journal =  {IEEE T. Pattern Analysis and Machine Intelligence (TPAMI)},
  year    =  {2021}
}
```
