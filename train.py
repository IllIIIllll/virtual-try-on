# © 2020 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from mrcnn import t_shirt as t
from mrcnn import model as modellib

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', default='data/t_shirt_dataset')
    parser.add_argument('--weights', default='mrcnn/mask_rcnn_coco.h5')
    parser.add_argument('--logs', default='logs')
    parser.add_argument('--image')
    parser.add_argument('--video')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
    elif args.mode == 'splash':
        assert args.image or args.video,\
               'Provide --image or --video to apply color splash'

    print('Weights :', args.weights)
    print('Dataset :', args.dataset)
    print('Logs :', args.logs)

    if args.mode == 'train':
        config = t.TshirtConfig()
    else:
        class InferenceConfig(t.TshirtConfig()):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    if args.mode == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir=args.logs)

    print('Loading weights', args.weights)
    model.load_weights(
        args.weights,
        by_name=True,
        exclude=[
            'mrcnn_class_logits',
            'mrcnn_bbox_fc',
            'mrcnn_bbox',
            'mrcnn_mask'
        ]
    )

    if args.mode == 'train':
        t.train(model, args.dataset, config)
    elif args.mode == 'splash':
        t.detect_and_color_splash(
            model,
            image_path=args.image,
            video_path=args.video
        )
    else:
        print(f"'{args.command}' is not recognized. \nUse 'train' or 'splash'")

if __name__ == '__main__':
    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    main()