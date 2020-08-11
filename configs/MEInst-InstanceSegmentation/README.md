# Mask Encoding for Single Shot Instance Segmentation

Rufeng Zhang, Zhi Tian, Chunhua Shen, Mingyu You, Youliang Yan

[[`arXiv`](https://arxiv.org/abs/2003.11712)] [[`BibTeX`](#CitingMEInst)]

## Models

### COCO Instance Segmentation Baselines with [MEInst](https://arxiv.org/abs/2003.11712)

Name | inf. time | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:
[MEInst_R_50_1x_none](MEInst_R_50_1x_none.yaml) | 13 FPS | 39.5 | 30.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/v49Av8jn9hDkSAT/download)
[MEInst_R_50_1x](MEInst_R_50_1x.yaml) | 12 FPS | 40.1 | 31.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/MB7jJycGDvI7z0E/download)
[MEInst_R_50_3x](MEInst_R_50_3x.yaml) | 12 FPS | 43.6 | 34.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/1ID0DeuI9JsFQoG/download)
[MEInst_R_50_3x_512](MEInst_R_50_3x_512.yaml) | 19 FPS | 40.8 | 32.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/T5pNmMbTr4wsyTd/download)

*Inference time is measured on a NVIDIA 1080Ti with batch size 1.*

## Quick Start

1. Download the [matrix](https://cloudstor.aarnet.edu.au/plus/s/rOLg2frN3MCeWr9/download) file for mask encoding during training 
2. Symlink the matrix path to datasets/components/xxx.npz, e.g., 
   `coco/components/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz` 
3. Follow [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for install, train and inference

### Step by step for Mask Encoding (Optional)

  We recommend to directly download the [matrix](https://cloudstor.aarnet.edu.au/plus/s/rOLg2frN3MCeWr9/download) file and use it, as it can already handle most cases.
And we also provide tools to generate encoding matrix yourself.

Example:

* Generate encoding matrix

  `python adet/modeling/MEInst/LME/mask_generation.py`

* Evaluate the quality of reconstruction

  `python adet/modeling/MEInst/LME/mask_evaluation.py`

## <a name="CitingMEInst"></a>Citing MEInst

If you use MEInst, please use the following BibTeX entry.

```BibTeX
@inproceedings{zhang2020MEInst,
  title     =  {Mask Encoding for Single Shot Instance Segmentation},
  author    =  {Zhang, Rufeng and Tian, Zhi and Shen, Chunhua and You, Mingyu and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/).
