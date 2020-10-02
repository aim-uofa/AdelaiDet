## BlendMask instance detection

```
coco/
  thing_train2017/
    # thing class label maps for auxiliary semantic loss
lvis/
  thing_train/
    # semantic labels for LVIS
```

Run `python prepare_thing_sem_from_instance.py`, to extract semantic labels from instance annotations.

Run `python prepare_thing_sem_from_lvis.py`, to extract semantic labels from LVIS annotations.

## Text Recognition

- Totaltext training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/32p6xsdtu0keu2o6pb5aqhyjotnljxep.zip) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[code]](https://github.com/cs-chan/Total-Text-Dataset). 
- CTW1500 training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/6ui89vca7cbp15ysnxqg5r494ix7l6cu.zip) [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [[code]](https://github.com/Yuliang-Liu/Curve-Text-Detector).
- MLT [[dataset]](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs)

```
text/
  totaltext/
    annotations/
    train_images/
    test_images/
  mlt2017/
    annotations/train.json
    images/
    ...
  syntext1/
  syntext2/
  ...
  evaluation/
    gt_ctw1500.zip
    gt_totaltext.zip
```

To evaluate on Total Text and CTW1500, first download the zipped annotations with

```
mkdir evaluation
cd evaluation
wget -O gt_ctw1500.zip https://cloudstor.aarnet.edu.au/plus/s/xU3yeM3GnidiSTr/download
wget -O gt_totaltext.zip https://cloudstor.aarnet.edu.au/plus/s/SFHvin8BLUM4cNd/download
```

## Person In Context instance detection

```
pic/
  thing_train/
    # thing class label maps for auxiliary semantic loss
  annotations/
    train_person.json
    val_person.json
  image/
    train/
    ...
  
```

First link the PIC_2.0 dataset to this folder with `ln -s \path\to\PIC_2.0 pic`. Then use the `python gen_coco_person.py` to generate train and validation annotation jsons.

Run `python prepare_thing_sem_from_instance.py --dataset-name pic`, to extract semantic labels from instance annotations.
