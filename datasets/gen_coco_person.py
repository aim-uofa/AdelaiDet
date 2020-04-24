import numpy as np
import cv2
import os
import json
error_list = ['23382.png', '23441.png', '20714.png', '20727.png', '23300.png', '21200.png']

def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos))
    return (x1, y1, x2, y2)

def gen_coco(phase):
    result = {
        "info": {"description": "PIC2.0 dataset."},
        "categories": [
            {"supercategory": "none", "id": 1, "name": "person"}
        ]
    }
    out_json = phase +'_person.json'
    store_segmentation = True

    images_info = []
    labels_info = []
    img_id = 0
    files = tuple(open("pic/list5/"+phase+'_id', 'r'))
    files = (_.strip() for _ in files)

    for index, image_name in enumerate(files):
        image_name = image_name+".png"
        print(index, image_name)
        if image_name in error_list:
            continue
        instance = cv2.imread(os.path.join('instance', phase, image_name), flags=cv2.IMREAD_GRAYSCALE)
        semantic = cv2.imread(os.path.join('semantic', phase, image_name), flags=cv2.IMREAD_GRAYSCALE)
        # print(instance.shape, semantic.shape)
        h = instance.shape[0]
        w = instance.shape[1]
        images_info.append(
            {
                "file_name": image_name[:-4]+'.jpg',
                "height": h,
                "width": w,
                "id": index
            }
        )
        instance_max_num = instance.max()
        instance_ids = np.unique(instance)
        for instance_id in instance_ids:
            if instance_id == 0:
                continue
            instance_part = instance == instance_id
            object_pos = instance_part.nonzero()
            # category_id_ = int(semantic[object_pos[0][0], object_pos[1][0]])
            category_id = int(np.max(semantic[object_pos[0], object_pos[1]]))
            # assert category_id_ == category_id, (category_id_, category_id)
            if category_id != 1:
                continue
            area = int(instance_part.sum())
            x1, y1, x2, y2 = mask2box(instance_part)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            segmentation = []
            if store_segmentation:
                contours, hierarchy = cv2.findContours((instance_part * 255).astype(np.uint8), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        segmentation.append(contour)
                if len(segmentation) == 0:
                    print('error')
                    continue
            labels_info.append(
                {
                    "segmentation": segmentation,  # poly
                    "area": area,  # segmentation area
                    "iscrowd": 0,
                    "image_id": index,
                    "bbox": [x1, y1, w, h],
                    "category_id": category_id,
                    "id": img_id
                },
            )
            img_id += 1
        # break
    result["images"] = images_info
    result["annotations"] = labels_info
    with open('pic/annotations/' + out_json, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    if not os.path.exists('pic/annotations/'):
        os.mkdirs('pic/annotations/')
    gen_coco("train")
    gen_coco("val")
    #gen_coco("test")
