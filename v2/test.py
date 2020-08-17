from inpaint_model import InpaintGenerator
from PIL import Image
import yaml
import argparse

import cv2
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = yaml.load(open('inpaint.yml', 'r'), Loader=yaml.FullLoader)
    args, unknown = parser.parse_known_args()

    image = np.array(Image.open(args.image).convert('RGB'))
    mask = np.array(Image.open(args.mask).convert('L'))
    mask = np.expand_dims(mask, 2)
    print(mask.shape, image.shape)

    assert image.shape[:2] == mask.shape[:2]

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0).astype(np.float32)
    mask = np.expand_dims(mask, 0).astype(np.float32)
    xin = image / 255.0
    mask = mask / 255.0

    G = InpaintGenerator()
    G.load_weights('logs/20200729-160205/models/G')

    xin = xin*(1.-mask)
    x1, x2, offset_flow = G(xin, mask, training=False)
    batch_predicted = x2
    result = batch_predicted * mask + xin * (1. - mask)
    cv2.imwrite(args.output, np.array(result[0][:, :, ::-1] * 255))
