import os
import sys
import logging
import shutil
import argparse

import cv2
import numpy as np
import tensorflow as tf

import config


app_name = os.path.splitext(os.path.basename(__file__))[0]
logger = config.get_logger(app_name)



def find_faces(account_path):
    
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
        face_image_resized = cv2.resize(face_image, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_AREA)    
        return face_image_resized

    logger.info(f'Handling account \'{os.path.basename(account_path)}\'')
    
    found_faces = []

    file_paths = config.get_paths(account_path)
    logger.info(f'Found {len(file_paths)} photos in account \'{os.path.basename(account_path)}\'')

    for image_path in config.get_paths(account_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(config.IMG_MIN_WIDTH_FIND, config.IMG_MIN_HEIGHT_FIND)
        )
        for face in faces:
            found_faces.append((get_resized_face(face, image, height, width), image_path))
    
    logger.info(f'Found {len(found_faces)} faces in account \'{os.path.basename(account_path)}\'')
    return found_faces

def get_prediction(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [config.IMG_HEIGHT, config.IMG_WIDTH])
    image_tensor /= 255.0
    image_tensor = np.reshape(image_tensor, [1, config.IMG_HEIGHT, config.IMG_WIDTH, 3])
    return model.predict_classes(image_tensor)[0]

def move_account(account_path, prediction_status, prediction_class):
    try:
        if prediction_status == 'FAILED':
            shutil.move(account_path, config.FAILED_PATH)
        elif prediction_status == 'SUCCESS':
            category_path = os.path.join(config.SUCCESS_PATH, config.CATEGORIZER_CLASSES[prediction_class])
            #if not os.path.exists(category_path):
            os.makedirs(category_path, exist_ok=True)
            shutil.move(account_path, category_path)
    except Exception as e:
        logger.error(f'Eror while moving account {e}')




def write_statistics(account_path, statistics):
    summary = sum(statistics)
    statistics_percentage = map(lambda x : 100 * x / sum(statistics), statistics)

    with open(os.path.join(account_path, 'INFO'), 'w+') as f:
        for pair in zip(config.CATEGORIZER_CLASSES, statistics_percentage):
            f.write(f'{pair[0].3f} - {pair[1]}%\n')



def predict(binary_model_path, categorizer_model_path):

    binary_filter = tf.keras.models.load_model(binary_model_path)
    categorizer = tf.keras.models.load_model(categorizer_model_path)

    accounts_paths = config.get_paths(config.ACCOUNTS_PATH)
    logger.info(f'Found {len(accounts_paths)} girls in {config.ACCOUNTS_PATH}')

    for account_path in accounts_paths:
        faces = find_faces(account_path)

        lit_counter = 0
        predictions_statistics = [0] * len(config.CATEGORIZER_CLASSES)
        
        for face in faces:
            if config.BINARY_FILTER_CLASSES[get_prediction(binary_filter, face[0])[0]] == 'LIT':
                logger.warning(f'Face \'{face[1]}\' is lit')
                lit_counter += 1
                continue
            
            predictions_statistics[get_prediction(categorizer, face[0])] += 1

        if lit_counter == len(faces):
            logger.warning(f'Move account \'{os.path.basename(account_path)}\' to FAILED')
            move_account(account_path, 'FAILED')
        else:
            intensified_prediction = predictions_statistics.index(max(predictions_statistics))
            logger.info(predictions_statistics)
            write_statistics(account_path, predictions_statistics)

            logger.info(f'Account \'{os.path.basename(account_path)}\' belongs to \'{config.CATEGORIZER_CLASSES[intensified_prediction]}\' class')
            move_account(account_path, 'SUCCESS', intensified_prediction)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--binary-filter-model', '-bfm', help='Path to binary-filter model', type=str, required=True)
    parser.add_argument('--categorizer-model', '-cm', help='Path to categorizer model', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    binary_model_path = os.path.join(config.ROOT_PATH, 'binary_filter/models/binary_filter_model_1587241639.h5')
    categorizer_model_path = os.path.join(config.ROOT_PATH, 'categorizer/models/my_model.h5')

    # predict(args.binary_filter_model, categorizer_model)
    predict(binary_model_path, categorizer_model_path)

if __name__ == '__main__':
    main()

