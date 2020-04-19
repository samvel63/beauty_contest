import os
import sys
import logging

import cv2
import numpy as np
import tensorflow as tf


root_path = os.path.dirname(os.path.abspath(__file__))
accounts_path = os.path.join(root_path, 'ACCOUNTS')
success_path = os.path.join(root_path, 'SUCCESS')
failed_path = os.path.join(root_path, 'FAILED')

binary_model_path = os.path.join(root_path, 'binary_filter/models/binary_filter_model_1587241639.h5')
categorizer_model_path = os.path.join(root_path, 'categorizer/models/my_model.h5')

image_size_width = 150
image_size_height = 150

IMG_HEIGHT = 150
IMG_WIDTH = 150


binary_filter_classes = ['GIRL', 'LIT']
categorizer_classes = ['ASIAN', 'BRUNETTE', 'DREADLOCK', 'GINGER', 'MULATTO']




app_name = os.path.splitext(os.path.basename(__file__))[0]

logger = logging.getLogger(app_name)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)



def get_paths(path):
    return sorted(os.path.join(path, file) for file in os.listdir(path))


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
        face_image_resized = cv2.resize(face_image, (image_size_width, image_size_height), interpolation=cv2.INTER_AREA)    
        return face_image_resized

    logger.info(f'Handling account \'{os.path.basename(account_path)}\'')
    
    found_faces = []

    file_paths = get_paths(account_path)
    logger.info(f'Found {len(file_paths)} photos in account \'{os.path.basename(account_path)}\'')

    for image_path in get_paths(account_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(80, 80)
        )
        for face in faces:
            found_faces.append(get_resized_face(face, image, height, width))
    
    logger.info(f'Found {len(found_faces)} faces in account \'{os.path.basename(account_path)}\'')
    return found_faces



def get_prediction(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [IMG_HEIGHT, IMG_WIDTH])
    image_tensor /= 255.0
    image_tensor = np.reshape(image_tensor, [1, IMG_HEIGHT, IMG_WIDTH, 3])
    return model.predict_classes(image_tensor)[0]


def main():

    binary_filter = tf.keras.models.load_model(binary_model_path)
    binary_filter.summary()

    categorizer = tf.keras.models.load_model(categorizer_model_path)
    categorizer.summary()

    accounts_paths = get_paths(accounts_path)
    logger.info(f'Found {len(accounts_paths)} girls')

    for account_path in accounts_paths:
        faces = find_faces(account_path)

        lit_counter = 0
        predictions_statistics = [0] * len(categorizer_classes)
        
        for face in faces:
            if binary_filter_classes[get_prediction(binary_filter, face)[0]] == 'LIT':
                lit_counter += 1
                continue
            
            predictions_statistics[get_prediction(categorizer, face)] += 1

        if lit_counter == len(faces):
            logger.info(f'Move account \'{os.path.basename(account_path)}\' to FAILED')
            # move_account(account_path, 'FAILED')
        else:
            intensified_prediction = predictions_statistics.index(max(predictions_statistics))
            logger.info(f'Account \'{os.path.basename(account_path)}\' belongs to \'{categorizer_classes[intensified_prediction]}\' class')

            #move_account(account_path, 'SUCCESS', intensified_prediction)




if __name__ == '__main__':
    main()
