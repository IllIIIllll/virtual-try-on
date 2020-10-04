# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from mrcnn.config import Config
from mrcnn import utils

import os
import datetime

import numpy as np
import skimage.draw
import cv2

COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'

if not os.path.exists('../logs'):
    os.mkdir('../logs')
DEFAULT_LOGS_DIR = '../logs'

class TshirtsConfig(Config):
    NAME = 'Tshirt'
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 200
    DETECTION_MIN_CONFIDENCE = 0.98

class TshirtsDataset(utils.Dataset):
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'Tshirt':
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros(
            [info['height'], info['width'], len(info['polygons'])],
            dtype=np.uint8
        )
        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'Tshirt':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)

def get_foreground_background(image, mask):
    if mask.shape[-1] > 0:
        fore_mask = (np.sum(mask, -1, keepdims=True) >= 1)
        back_mask = (np.sum(mask, -1, keepdims=True) < 1)

        foreground = np.where(fore_mask, image, 255).astype(np.uint8)
        background = np.where(back_mask, image, 0).astype(np.uint8)
    else:
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        foreground = gray.astype(np.uint8)
        background = gray.astype(np.uint8)
    return foreground, background

def crop_and_pad(image_in, image_out, bbox):
    if (bbox[2] - bbox[0]) % 2 != 0:
        bbox[0] += 1
    if (bbox[3] - bbox[1]) % 2 != 0:
        bbox[1] += 1

    y1, x1, y2, x2 = bbox
    crop_img = image_in[y1:y2, x1:x2]

    w = x2 - x1
    h = y2 - y1

    if w >= h:
        p = int((w - h) / 2)
        img_padding = cv2.copyMakeBorder(
            crop_img,
            p,
            p,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    else:
        p = int((h - w) / 2)
        img_padding = cv2.copyMakeBorder(
            crop_img,
            0,
            0,
            p,
            p,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    cv2.imwrite(image_out, cv2.cvtColor(img_padding, cv2.COLOR_RGB2BGR))
    return bbox

def get_mask_save_segimage(model, image_path,
                           fore_file_path, back_file_path, save_back=True):
    print(f'Running on {image_path}')
    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)[0]

    fore, back = get_foreground_background(image, r['masks'])

    if save_back:
        skimage.io.imsave(back_file_path, back)

    bbox = crop_and_pad(fore, fore_file_path, r['rois'][0])

    print('Foreground Saved to ', fore_file_path)
    print('Background Saved to ', back_file_path)
    return r['masks'], bbox

def user_style_seg(user_input, style_input, model, weight, output_dir):
    user_fore = output_dir + f'user_foreground_{datetime.datetime.now():%Y%m%dT%H%M%S}'
    user_back = output_dir + f'user_background_{datetime.datetime.now():%Y%m%dT%H%M%S}'
    style_fore = output_dir + f'style_foreground_{datetime.datetime.now():%Y%m%dT%H%M%S}'
    style_back = output_dir + f'style_background_{datetime.datetime.now():%Y%m%dT%H%M%S}'

    user_mask, user_bbox = get_mask_save_segimage(model, user_input,
                                                  user_fore, user_back, save_back=True)
    _, _ = get_mask_save_segimage(model, style_input,
                                  style_fore, style_back, save_back=False)

    return user_fore, user_back, style_fore, user_mask, user_bbox

def image_rendering(tshirt, background, user_bbox, user_mask, output_dir):
    t = cv2.imread(tshirt, cv2.IMREAD_COLOR)
    bg = cv2.imread(background, cv2.IMREAD_COLOR)
    bg_h, bg_w, _ = bg.shape

    y1, x1, y2, x2 = user_bbox
    w = x2 - x1
    h = y2 - y1

    if w >= h:
        p = int((w - h) / 2)
        t_resized = cv2.resize(t, (h, h), interpolation=cv2.INTER_AREA)
        t_crop = t_resized[p:p + h, :]
    else:
        p = int((h - w) / 2)
        t_resized = cv2.resize(t, (h, h), interpolation=cv2.INTER_AREA)
        t_crop = t_resized[:, p:p + w]

    t_padding = cv2.copyMakeBorder(t_crop, y1, bg_h - y2, x1, bg_w - x2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    _, back = get_foreground_background(bg, user_mask)
    fore, _ = get_foreground_background(t_padding, user_mask)
    fore = cv2.cvtColor(fore, cv2.COLOR_RGB2BGR)
    out = np.where(back == [0, 0, 0], fore, back).astype(np.uint8)

    out_path = output_dir + f'final_output_{datetime.datetime.now():%Y%m%dT%H%M%S}'
    cv2.imwrite(out_path, out)
