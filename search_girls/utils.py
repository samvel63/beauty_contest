import os
import logging


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
    x, y, w, h = (v for v in face)
    x_inc = int(w*0.35)
    y_inc = int(h*0.35)

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
    face_image_resized = cv2.resize(face_image, (utils.IMG_WIDTH, utils.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return face_image_resized
