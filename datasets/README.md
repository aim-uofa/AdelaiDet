## AdelaiDet instance detection

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
- MLT [[images]](https://universityofadelaide.box.com/s/9c4maycxaxo6dd95sfjz087pno3wbnm7)[[annos]](https://universityofadelaide.box.com/shared/static/8hgcrfdvqroqjwy27thu1naez6px82a1.zip) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
- Syntext-150k (Part1: 54,327 [[imgs]](https://universityofadelaide.box.com/s/1jcvu6z9jojmhzojuqrwxvwxmrlw7uib)[[annos]](https://universityofadelaide.box.com/s/zc73pyzvymqkjg3vkb2ayjol7y5a4fsk).
- Part2: 94,723 [[imgs]](https://universityofadelaide.box.com/s/ibihmhkzpc1zuh56mxyehad1dv1l73ua)[[annos]](https://universityofadelaide.box.com/s/rk55zheij8ubvwgzg7dfjbxgi27l8xld).) 

```
text/
  totaltext/
    annotations/
    train_images/
    test_images/
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
