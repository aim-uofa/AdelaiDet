# [VoVNet-v2](https://github.com/youngwanLEE/CenterMask) backbone networks in [FCOS](https://github.com/aim-uofa/adet)
**Efficient Backbone Network for Object Detection and Segmentation**\
Youngwan Lee


[[`vovnet-detectron2`](https://github.com/youngwanLEE/vovnet-detectron2)][[`CenterMask(code)`](https://github.com/youngwanLEE/CenterMask)] [[`VoVNet-v1(arxiv)`](https://arxiv.org/abs/1904.09730)] [[`VoVNet-v2(arxiv)`](https://arxiv.org/abs/1911.06667)] [[`BibTeX`](#CitingVoVNet)]


<div align="center">
  <img src="https://dl.dropbox.com/s/jgi3c5828dzcupf/osa_updated.jpg" width="700px" />
</div>

  
## Comparison with Faster R-CNN and ResNet

### Note

We measure the inference time of all models with batch size 1 on the same V100 GPU machine.

- pytorch1.3.1
- CUDA 10.1
- cuDNN 7.3


|Method|Backbone|lr sched|inference time|AP|APs|APm|APl|download|
|---|:--------:|:---:|:--:|--|----|----|---|--------|
|Faster|R-50-FPN|3x|0.047|40.2|24.2|43.5|52.0|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a>
|Faster|**V2-39-FPN**|3x|0.047|42.7|27.1|45.6|54.0|<a href="https://dl.dropbox.com/s/dkto39ececze6l4/faster_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dx9qz1dn65ccrwd/faster_V_39_eSE_ms_3x_metrics.json">metrics</a>
|**FCOS**|**V2-39-FPN**|3x|0.045|43.5|28.1|47.2|54.5|<a href="https://dl.dropbox.com/s/t51vrqiekid49vp/fcos_V_39_eSE_FPN_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/jhu301a95o7lzw1/fcos_V_39_eSE_FPN_ms_3x_metrics.json">metrics</a>
||
|Faster|R-101-FPN|3x|0.063|42.0|25.2|45.6|54.6|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a>
|Faster|**V2-57-FPN**|3x|0.054|43.3|27.5|46.7|55.3|<a href="https://dl.dropbox.com/s/c7mb1mq10eo4pzk/faster_V_57_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/3tsn218zzmuhyo8/faster_V_57_eSE_metrics.json">metrics</a>
|**FCOS**|**V2-57-FPN**|3x|0.051|44.4|28.8|47.2|56.3|<a href="https://dl.dropbox.com/s/c7mb1mq10eo4pzk/faster_V_57_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/3tsn218zzmuhyo8/faster_V_57_eSE_metrics.json">metrics</a>
||
|Faster|X-101-FPN|3x|0.120|43.0|27.2|46.1|54.9|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a>|
|Faster|**V2-99-FPN**|3x|0.073|44.1|28.1|47.0|56.4|<a href="https://dl.dropbox.com/s/v64mknwzfpmfcdh/faster_V_99_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/zvaz9s8gvq2mhrd/faster_V_99_eSE_ms_3x_metrics.json">metrics</a>|
|**FCOS**|**V2-99-FPN**|3x|0.070|45.2|29.2|48.4|57.3|<a href="https://www.dropbox.com/s/cztd5jry52cy6vx/fcos_V_99_eSE_FPN_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/zdfb5zjl9lhi5p8/fcos_V_99_eSE_FPN_ms_3x_metrics.json">metrics</a>|



## <a name="CitingVoVNet"></a>Citing VoVNet

If you use VoVNet, please use the following BibTeX entry.

```BibTeX
@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2019}
}

@article{lee2019centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  journal={arXiv preprint arXiv:1911.06667},
  year={2019}
}
```
