# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from vton import utils

from os.path import join as opj
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--category', default=1)
    parser.add_argument('--mode', default='segmentation')

    args = parser.parse_args()

    original_img = opj('data', args.data, 'image')
    original_anno = opj('data', args.data, 'annos')
    selected_img = opj('data/t_shirt_dataset', args.data, 'image')
    selected_anno = opj('data/t_shirt_dataset', args.data, 'annos')

    utils.t_shirt_img_copy(
        original_anno,
        original_img,
        selected_img,
        args.category
    )

    utils.anno_copy_by_img(
        selected_img,
        original_anno,
        selected_anno
    )

    utils.coco_to_via(
        selected_anno,
        selected_img,
        args.category,
        opj('data/t_shirt_dataset', args.data),
        args.mode,
    )

if __name__ == '__main__':
    main()