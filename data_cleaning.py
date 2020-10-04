# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from vton import utils

import os
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--original-path-img',
        default='data/validation/image',
    )
    parser.add_argument(
        '--original-path-anno',
        default='data/validation/annos',
    )
    parser.add_argument(
        '--original-train-img',
        default='data/train/image',
    )
    parser.add_argument(
        '--original-train-anno',
        default='data/train/annos',
    )
    parser.add_argument(
        '--selected-img-path',
        default='data/t_shirts_dataset/all_images',
    )
    parser.add_argument(
        '--selected-anno-path',
        default='data/t_shirts_dataset/all_annos',
    )
    parser.add_argument(
        '--img-json-path',
        default='data/t_shirts_dataset/division_Tshirts.json',
    )
    parser.add_argument(
        '--t-val-anno',
        default='data/t_shirts_dataset/t_val_anno',
    )
    parser.add_argument(
        '--t-val-img',
        default='data/t_shirts_dataset/t_val_img',
    )
    parser.add_argument(
        '--t-train-anno',
        default='data/t_shirts_dataset/t_train_anno',
    )
    parser.add_argument(
        '--t-train-img',
        default='data/t_shirts_dataset/t_train_img',
    )
    parser.add_argument(
        '--train-path-img',
        default='data/t_shirts_dataset/train/img',
    )
    parser.add_argument(
        '--train-path-anno',
        default='data/t_shirts_dataset/train/anno',
    )
    parser.add_argument(
        '--val-path-img',
        default='data/t_shirts_dataset/val/img',
    )
    parser.add_argument(
        '--val-path-anno',
        default='data/t_shirts_dataset/val/anno',
    )

    args = parser.parse_args()

    utils.anno_copy_by_img(
        args.t_train_img,
        args.selected_anno_path,
        args.t_train_anno
    )
    utils.anno_copy_by_img(
        args.t_val_img,
        args.selected_anno_path,
        args.t_val_anno
    )

    t_shirts = utils.contains_category_filenames(
        args.original_train_anno,
        category_id=1
    )

    for data in t_shirts:
        shutil.copy(
            os.path.join(
                args.original_train_img,
                data[0].replace('.json', '.jpg')
            ),
            os.path.join(
                args.selected_img_path,
                data[0].replace('.json', '.jpg')
            )
        )

    utils.not_contains_category_filenames(
        args.selected_anno_path,
        category_id=1
    )
    utils.COCO_to_VIA(
        anno_dir=args.t_train_anno,
        image_dir=args.t_train_img,
        category_id=1,
        save_anno_dir=args.t_train_img
    )
    utils.COCO_to_VIA(
        anno_dir=args.t_val_anno,
        image_dir=args.t_val_img,
        category_id=1,
        save_anno_dir=args.t_val_img
    )

    utils.anno_copy_by_img(
        args.train_path_img,
        args.original_path_anno,
        args.train_path_anno
    )
    utils.COCO_to_VIA(
        anno_dir=args.train_path_anno,
        image_dir=args.train_path_img,
        category_id=1,
        save_anno_dir=args.train_path_anno,
        mode="segmentation"
    )

    utils.anno_copy_by_img(
        args.val_path_img,
        args.original_path_anno,
        args.val_path_anno
    )
    utils.COCO_to_VIA(
        anno_dir=args.val_path_anno,
        image_dir=args.val_path_img,
        category_id=1,
        save_anno_dir=args.val_path_anno,
        mode="segmentation"
    )

    utils.img_filenames_to_json(
        args.selected_img_path,
        args.img_json_path
    )
    utils.img_copy_by_json(
        args.original_train_img,
        args.selected_img_path,
        args.img_json_path
    )

if __name__ == '__main__':
    main()