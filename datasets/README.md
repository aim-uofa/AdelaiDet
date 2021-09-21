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

English pretrained data:

- Totaltext training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/32p6xsdtu0keu2o6pb5aqhyjotnljxep.zip) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[code]](https://github.com/cs-chan/Total-Text-Dataset). 
- CTW1500 training, testing images, and annotations [[link]](https://universityofadelaide.box.com/shared/static/6ui89vca7cbp15ysnxqg5r494ix7l6cu.zip) [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [[code]](https://github.com/Yuliang-Liu/Curve-Text-Detector).
- MLT [[dataset]](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m) [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs)
  - If you have trouble downloading Syntext-150k, you can try BaiduNetDisk [[here]](https://github.com/aim-uofa/AdelaiDet/issues/312)

Chinese pretrained data:

- ReCTs [[images&label]](https://drive.google.com/file/d/1ygDN1OHUusqzqJL2011wc2T_LX0t6Th4/view?usp=sharing)(1.7G) [[Origin_of_dataset]](https://rrc.cvc.uab.es/?ch=12)
- ReCTs test set [[images&empty_label]](https://drive.google.com/file/d/1WEvkLgFIWdEDQn2UXHKCTIqlNnlk4kVt/view?usp=sharing)(0.5G) [[Origin_of_dataset]](https://rrc.cvc.uab.es/?ch=12)
- LSVT [[images&label]](https://drive.google.com/file/d/1E9RMFiRaRW4WdzA9Py7OimfzA82-Bwik/view?usp=sharing)(8.2G) [[Origin_of_dataset]](https://rrc.cvc.uab.es/?ch=16)
- ArT [[images&label]](https://drive.google.com/file/d/1ss_3oYVYexSmhx7AP4cahl8Emd49Wrh8/view?usp=sharing)(1.5G)  [[Origin_of_dataset]](https://rrc.cvc.uab.es/?ch=14)
- SynChinese130k [[images&label]](https://drive.google.com/file/d/1w9BFDTfVgZvpLE003zM694E0we4OWmyP/view?usp=sharing)(25G) [[Origin_of_dataset]](https://arxiv.org/abs/2105.03620)
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
