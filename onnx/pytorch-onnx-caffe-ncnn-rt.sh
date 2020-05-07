
if [ "$1" == "update" ];
then
  update='update'
else
  update=
fi

if [ "$1" == "test" ];
then
  onnx_rt='test'
else
  onnx_rt=
fi

caffe_repo=/data/pretrained/caffe/fcos/
onnx_repo=/data/pretrained/onnx/fcos/
pytorch_repo=/data/pretrained/pytorch/fcos/
ncnn_repo=/data/pretrained/ncnn/fcos/

config=configs/FCOS-Detection/FCOS_RT/MS_R_50_4x_syncbn_bn_head.yaml
case=FCOS_RT_MS_R_50_4x_syncbn_bn_head

if [ ! -e $onnx_repo/$case.onnx ] || [ "$update" != "" ];
then
  cd /workspace/git/uofa-AdelaiDet/ # folder of project https://github.com/aim-uofa/AdelaiDet
  pwd
  python -V # ensure python3.x
  python onnx/export_model_to_onnx.py \
    --config-file $config \
    --output $onnx_repo/$case.onnx \
    --width 768 --height 640 \
    --opts MODEL.WEIGHTS $pytorch_repo/$case.pth MODEL.FCOS.NORM "BN" MODEL.DEVICE cpu
  if [ $? -ne 0 ]; then exit; fi
fi

if [ ! -e $onnx_repo/$case-update.onnx ] || [ "$update" != "" ];
then
  # advise version 1.3.0
  cd /workspace/git/onnx-simplifier  # folder of project: https://github.com/daquexian/onnx-simplifier
  pwd
  python -V # ensure python3.x
  python -m onnxsim $onnx_repo/$case.onnx $onnx_repo/$case-update.onnx
  if [ $? -ne 0 ]; then exit; fi
fi

# optional
if [ ! -e $caffe_repo/$case-update.caffemodel ];
then
  # switch to python2 and ensure caffe (with the upsample patch) ready
  # refer: https://github.com/blueardour/caffe.git  for patched version
  cd /workspace/git/onnx2caffe  # folder of project: https://github.com/MTlab/onnx2caffe
  pwd
  python -V
  python convertCaffe.py $onnx_repo/$case-update.onnx $caffe_repo/$case-update.prototxt $caffe_repo/$case-update.caffemodel
  if [ $? -ne 0 ]; then exit; fi
fi

# ncnn
if [ ! -e $ncnn_repo/$case-update-opt.bin ] || [ "$update" != "" ]
then
  cd /workspace/git/ncnn # folder of project: https://github.com/Tencent/ncnn
  pwd
  mkdir -p $ncnn_repo
  ./build-host-gcc-linux/tools/onnx/onnx2ncnn $onnx_repo/$case-update.onnx $ncnn_repo/$case-update.param $ncnn_repo/$case-update.bin
  if [ $? -eq 0 ]; then
    echo "Optimizing"
    ./build-host-gcc-linux/tools/ncnnoptimize $ncnn_repo/$case-update.param $ncnn_repo/$case-update.bin \
      $ncnn_repo/$case-update-opt.param $ncnn_repo/$case-update-opt.bin \
      0 #data 640 512 3
  else
    echo "Convert failed"
  fi
fi

if [ "$onnx_rt" == "test" ];
then
  cd /workspace/git/uofa-AdelaiDet/ # folder of project https://github.com/aim-uofa/AdelaiDet
  pwd
  python -V # ensure python3.x
  python onnx/test_onnxruntime.py \
    --config-file $config \
    --output $onnx_repo/$case.onnx \
    --width 768 --height 640 \
    --opts MODEL.WEIGHTS $pytorch_repo/$case.pth MODEL.FCOS.NORM "BN" MODEL.DEVICE cpu
  if [ $? -ne 0 ]; then exit; fi
fi


