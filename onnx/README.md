
# update history

2020.05.07:

1. add [onnxruntime](https://github.com/microsoft/onnxruntime) verification demo 

2. add RT model all-in-one script

# Export to onnx/caffe/ncnn

Refer all-in-one script: [pytorch-onnx-caffe-ncnn.sh](pytorch-onnx-caffe-ncnn.sh) (BN instead of GN used in the FCOS head)

Refer another all-in-one script: [pytorch-onnx-caffe-ncnn-rt.sh](pytorch-onnx-caffe-ncnn-rt.sh) for the RT model alone with onnxruntime verification demo

note: to convert model to *Caffe* and *NCNN* requires BN in the FCOS head

# Normalization in the FCOS head
The normalization in FOCS head is GroupNorm (GN) by default as shown in the original paper. Unlike BN, GN caculates the mean and variance of features online. Thus, it costs extra time and memory.
On the other hand, as BN can be merged into the previous convolution layer,  BN introduces no computation overhead during inference. The following instruction introduces a simple method to measure the impact of GN on speed.

* prepare some images (for example 1000) in folder output/test/input/

* include time measurement code in demo/demo.py

* GN + GPU: total execution time 285.1398s, average 0.0696s per image

```sh
python demo/demo --config-file configs/FCOS-Detection/R_50_1x.yaml --input output/test/input/ --output output/test/output/  --opts MODEL.WEIGHTS weights/fcos_R_50_1x.pth
```
* BN + GPU: total execution time 257.4333s, average 0.0628s per image
```sh
python demo/demo.py --config-file configs/FCOS-Detection/R_50_1x.yaml --input output/test/input/ --output output/test/output/  --opts MODEL.WEIGHTS weights/fcos_R_50_1x.pth MODEL.FCOS.NORM BN
```
* GN + CPU: total execution time 1125.4375s, average 1.0112s per image
```sh
python demo/demo.py --config-file configs/FCOS-Detection/R_50_1x.yaml --input output/test/input/ --output output/test/output/  --opts MODEL.WEIGHTS weights/fcos_R_50_1x.pth MODEL.DEVICE cpu
```
* BN + CPU: total execution time 1068.0550s, average 0.9596s per image
```sh
python demo/demo.py --config-file configs/FCOS-Detection/R_50_1x.yaml --input output/test/input/ --output output/test/output/  --opts MODEL.WEIGHTS weights/fcos_R_50_1x.pth MODEL.DEVICE cpu MODEL.FCOS.NORM BN
```

Tested on 2080ti. The result shows 5~10% slower for GN compared against BN.

# Result compare between pytorch and NCNN

* pytorch version: run demo/demo.py
* ncnn version: refer to https://github.com/blueardour/ncnn/blob/master/examples/fcos.cpp

Example: take coco/test2017/000000144041.jpg as the test image

```
#> cd AdelaiDet

#> mkdir -p output/test/

#> cp $COCO/test2017/000000144041.jpg output/test/input.jpg

#> python demo/demo.py --config-file configs/FCOS-Detection/R_50_1x.yaml --input output/test/input.jpg --output output/test/output.jpg --opts MODEL.WEIGHTS /data/pretrained/pytorch/fcos/FCOS_R_50_1x_bn_head.pth MODEL.FCOS.NORM "BN" MODEL.DEVICE cpu


#> cd $NCNN_ROOT  # (build the project ahead)

#> cd build-host-gcc-linux/examples

#> ln -s /data/pretrained/ncnn/fcos/FCOS_R_50_1x_bn_head-update-opt.bin net.bin  # (refer pytorch-onnx-caffe-ncnn.sh to generate the file)

#> ln -s /data/pretrained/ncnn/fcos/FCOS_R_50_1x_bn_head-update-opt.param net.param  (refer pytorch-onnx-caffe-ncnn.sh to generate the file)

#> ./fcos /workspace/git/uofa-AdelaiDet/output/test/input.jpg net.param net.bin 800 1088
```

