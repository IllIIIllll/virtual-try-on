# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
from os.path import join as opj
import json
import shutil

import cv2

def anno_copy_by_img(selected_img, ori_anno_dir, selected_anno):
    if not os.path.exists(selected_anno):
        os.makedirs(selected_anno)

    img_data = os.listdir(selected_img)
    json_data = []

    for img_file_name in img_data:
        json_data.append(img_file_name.replace(".jpg", ".json"))

    for json_file_name in json_data:
        shutil.copy(
            opj(ori_anno_dir, json_file_name),
            opj(selected_anno, json_file_name)
        )

def t_shirt_img_copy(ori_anno_dir, ori_img_dir, selected_img, category_id):
    if not os.path.exists(selected_img):
        os.makedirs(selected_img)

    t_shirt = contains_category(ori_anno_dir, category_id=category_id)

    for data in t_shirt:
        shutil.copy(
            os.path.join(ori_img_dir, data[0].replace('.json', '.jpg')),
            os.path.join(selected_img,data[0].replace('.json', '.jpg'))
        )

def contains_category(anno_dir, category_id):
    contains = []
    MAX_FILES = 10000

    if not os.path.exists(anno_dir):
        print('anno_dir not exists')
        return

    directory = os.listdir(anno_dir)
    directory = directory[:MAX_FILES]

    for anno in directory:
        with open(opj(anno_dir, anno)) as json_file:
            json_data = json.load(json_file)
        if json_data['item1']['category_id'] != category_id:
            try:
                if not json_data['item2']['category_id'] != category_id:
                    print(anno, 'item 2 :', json_data['item2']['category_name'])
                    contains.append([anno, 2])

            except:
                pass
        else:
            print(anno, 'item 1 :', json_data['item1']['category_name'])
            contains.append([anno, 1])
    return contains

def seg_to_points(segmentation):
    hl = len(segmentation) // 2
    x = [segmentation[i * 2] for i in range(hl)]
    y = [segmentation[i * 2 + 1] for i in range(hl)]
    return x, y

def lm_to_points(landmarks):
    hl = len(landmarks) // 3
    x = [landmarks[i * 3] for i in range(hl)]
    y = [landmarks[1 + i * 3] for i in range(hl)]
    return x, y

def coco_to_via(anno_dir, image_dir, category_id, save_anno_dir, mode="segmentation"):
    if not os.path.exists(anno_dir):
        print('anno_dir not exists')
        return
    if not os.path.exists(image_dir):
        print('image_dir not exists')
        return
    if not os.path.exists(save_anno_dir):
        os.makedirs(save_anno_dir)

    directory = os.listdir(anno_dir)

    VIA_dict = {}

    json_file_name = 'via_region_data.json'

    for anno in directory:
        with open(opj(anno_dir, anno)) as json_file:
            json_data = json.load(json_file)

        img_path = anno.replace('.json', '.jpg')

        img_abs_path = opj(image_dir, img_path)
        image = cv2.imread(img_abs_path)
        height, width = image.shape[:2]
        img_size = height * width

        if json_data['item1']['category_id'] != category_id:
            try:
                if json_data['item2']['category_id'] != category_id:
                    print('Not contains short sleeve top :', anno)
                else:
                    if mode == "segmentation":
                        all_points_x, all_points_y = seg_to_points(
                            json_data['item2']['segmentation'][0]
                        )
                    elif mode == "landmarks":
                        all_points_x, all_points_y = lm_to_points(
                            json_data['item2']['landmarks']
                        )
                    VIA_dict[img_path] = {
                        "fileref": "",
                        "size": img_size,
                        "filename": img_path,
                        "base64_img_data": "",
                        "file_attributes": {},
                        "regions": {
                            "0": {
                                "shape_attributes": {
                                    "name": "polygon",
                                    "all_points_x": all_points_x,
                                    "all_points_y": all_points_y
                                },
                                "region_attributes": {}
                            }
                        }
                    }

            except:
                print('Not contains short sleeve top :',anno)
        else:
            if mode == "segmentation":
                all_points_x, all_points_y = seg_to_points(
                    json_data['item1']['segmentation'][0]
                )
            elif mode == "landmarks":
                all_points_x, all_points_y = lm_to_points(
                    json_data['item1']['landmarks']
                )
            VIA_dict[img_path] = {
                "fileref": "",
                "size": img_size,
                "filename": img_path,
                "base64_img_data": "",
                "file_attributes": {},
                "regions": {
                    "0": {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": all_points_x,
                            "all_points_y": all_points_y
                        },
                        "region_attributes": {}
                    }
                }
            }
    with open(opj(save_anno_dir, json_file_name), 'w') as f:
        json.dump(VIA_dict, f)