import os
import shutil
import logging

import cv2

from search_girls import configs


def get_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_paths(path):
    return sorted(os.path.join(path, file) for file in os.listdir(path))


def get_resized_face(face, image, height, width):
    x, y, w, h = face

    x_inc, y_inc = int(w*0.35), int(h*0.35)

    x0, x1, y0, y1 = x, x+w, y, y+h

    if x-x_inc > 0:
        x0 -= x_inc
    if x1+x_inc < width:
        x1 += x_inc
    if y-y_inc > 0:
        y0 -= y_inc
    if y1+y_inc < height:
        y1 += y_inc

    face_image = image[y0:(y1+y_inc), x0:(x1+x_inc)]
    face_image_resized = cv2.resize(face_image, (configs.IMG_WIDTH, configs.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return face_image_resized


def move_folder(logger, path_from, path_to):
    logger.info(f'Moving {path_from} to {path_to}')
    try:
        os.makedirs(path_to, exist_ok=True)
        shutil.move(path_from, path_to)
        logger.info('Successfully moved')
    except Exception as e:
        logger.error(f'failed while moving account with message: {e}')
