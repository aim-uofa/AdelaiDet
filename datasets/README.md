## Expected dataset structure for AdelaiDet instance detection:

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

## Expected dataset structure for Person In Context instance detection:

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
