from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

import numpy as np
import tensorflow as tf


CLASSES = ['asian', 'brunette', 'dreadlock', 'ginger', 'mulatto']

IMG_HEIGHT = 150
IMG_WIDTH = 150


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', '-mp', help='Path of model `{model_name}.h5`', required=True)
    parser.add_argument('--images-path', '-p', help='Path of photos', required=True)

    return parser.parse_args()


def get_paths(path):
    return sorted(os.path.join(path, file) for file in os.listdir(path))


def predict_image(model, image_path):
    image = tf.io.read_file(image_path)

    image_tensor = tf.image.decode_image(image)
    image_tensor = tf.image.resize(image_tensor, [IMG_HEIGHT, IMG_WIDTH])
    image_tensor /= 255.0

    image_tensor = np.reshape(image_tensor, [1, IMG_HEIGHT, IMG_WIDTH, 3])

    classes = model.predict_classes(image_tensor)
    print(image_path, CLASSES[classes[0]], classes[0])


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)
    model.summary()

    images_path = get_paths(args.images_path)
    for image_path in images_path:
        predict_image(model, image_path)


if __name__ == '__main__':
    main()
