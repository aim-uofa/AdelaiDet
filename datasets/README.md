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

- Totaltext training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/3eq5ti7z45qfq5gu96gg5t1xwh1yrrt7.zip) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[code]](https://github.com/cs-chan/Total-Text-Dataset). 
- CTW1500 training, testing images, and annotations [[link]](https://universityofadelaide.box.com/s/yb9red8pi9eszuzqompo593b6zhz87qw) [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [[code]](https://github.com/Yuliang-Liu/Curve-Text-Detector).
- MLT [[dataset]](https://universityofadelaide.box.com/s/tsiimvp65tkf7dw1nuh8l71cjcs0fyif) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/alta996w4fym6arh977h3k3xv55clhg3) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/7k7d6nvf951s4i01szs4udpu2yv5dlqe)

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
wget -O gt_ctw1500.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download
wget -O gt_totaltext.zip https://cloudstor.aarnet.edu.au/plus/s/pEMs0KjCocL2nvV/download
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
