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

## Text Spotting

English pretrained data:

- Totaltext [[paper]](https://ieeexplore.ieee.org/abstract/document/8270088/) [[source]](https://github.com/cs-chan/Total-Text-Dataset). 
  - Download (0.4G) ([Origin](https://universityofadelaide.box.com/shared/static/32p6xsdtu0keu2o6pb5aqhyjotnljxep.zip), [Google](https://drive.google.com/file/d/17JvGhzcbM54txG-lVRRDe6Eym8_ZM2uJ/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1nFAcqQciia4CvVmR8FRUsw) password: kgy7) 
  
- CTW1500 [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) [[source]](https://github.com/Yuliang-Liu/Curve-Text-Detector).
  - Download (0.8G) ([Origin](https://universityofadelaide.box.com/shared/static/6ui89vca7cbp15ysnxqg5r494ix7l6cu.zip), [Google](https://drive.google.com/file/d/1mBwRlMtFPFgR6QJW9F-rSBbc6N2cFvgZ/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1UrMl2qQNnChc2g2oyL4lcA) password: 7kvx)
   
- MLT [[paper]](https://ieeexplore.ieee.org/abstract/document/8270168).
  - Download (6.8G) ([Origin](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m), [Google](https://drive.google.com/file/d/1nE2d_sIfcAejgVIv6-UjGNcBXgxc4QfD/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1rjqmb3uuki_Ppcxq-tl7oQ) password: zqrm)
  
- CurvedSynText150k [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf): 
  - Part1 (94,723) Download (15.8G) ([Origin](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p), [Google](https://drive.google.com/file/d/1OSJ-zId2h3t_-I7g_wUkrK-VqQy153Kj/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1Y5pqVqfjcc4FKxW4y8R5jw) password: 4k3x) 
  - Part2 (54,327) Download (9.7G) ([Origin](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs), [Google](https://drive.google.com/file/d/1EzkcOlIgEp5wmEubvHb7-J5EImHExYgY/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1gRv-IjqAUu6qnXN5BXlOzQ) password: a5f5)

Chinese pretrained data:

- ReCTs [[Source]](https://rrc.cvc.uab.es/?ch=12)
  - Download (1.7G) ([Google](https://drive.google.com/file/d/1ygDN1OHUusqzqJL2011wc2T_LX0t6Th4/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1iZsnweBNJH3UNtGB5MCwKg) password: wo3o)
- ReCTs test set [[Source]](https://rrc.cvc.uab.es/?ch=12)
  - Download (0.5G) ([Google](https://drive.google.com/file/d/1WEvkLgFIWdEDQn2UXHKCTIqlNnlk4kVt/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1NTdULcWR14M8O_CmOsJz9w) password: l1zy)
- LSVT [[Source]](https://rrc.cvc.uab.es/?ch=16)
  - Download (8.2G) ([Google](https://drive.google.com/file/d/1E9RMFiRaRW4WdzA9Py7OimfzA82-Bwik/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1MTC5ZQno3KT65fDGoHENwA) password: qv7k)
- ArT [[Source]](https://rrc.cvc.uab.es/?ch=14)
  - Download (1.5G) ([Google](https://drive.google.com/file/d/1ss_3oYVYexSmhx7AP4cahl8Emd49Wrh8/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1XRLGzuGpxkNZXwmGy9jfbQ) password: ozht)
- SynChinese130k [[Source]](https://arxiv.org/abs/2105.03620)
  - Download (25G) ([Google](https://drive.google.com/file/d/1w9BFDTfVgZvpLE003zM694E0we4OWmyP/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1DYfTVkkz5bvAmqxDWMhFlA) password: zc3q)
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
