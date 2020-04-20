import os
import argparse

import cv2
import numpy as np
import tensorflow as tf

try:
    from search_girls import utils, configs
except ModuleNotFoundError as e:
    import utils
    import configs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-girls', '-mg', help='Number of predicted girls per one execute of the program',
                        type=int, required=True)

    parser.add_argument('--binary-filter-model', '-bfm', help='Path to binary-filter model', required=True)
    parser.add_argument('--categorizer-model', '-cm', help='Path to categorizer model', required=True)

    return parser.parse_args()


def find_faces(logger, account_path):
    logger.info(f'Handling account in path {account_path}')
    
    found_faces = []

    file_paths = utils.get_paths(account_path)
    logger.info(f'Found {len(file_paths)} photos in account path {account_path}')

    for image_path in utils.get_paths(account_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(configs.IMG_MIN_WIDTH_FIND, configs.IMG_MIN_HEIGHT_FIND)
        )
        for face in faces:
            resized_face = utils.get_resized_face(face, image, height, width)
            found_faces.append((resized_face, image_path))
    
    logger.info(f'Found {len(found_faces)} faces in account path {account_path}')

    return found_faces


def get_prediction(model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [configs.IMG_HEIGHT, configs.IMG_WIDTH])
    image_tensor /= 255.0
    image_tensor = np.reshape(image_tensor, [1, configs.IMG_HEIGHT, configs.IMG_WIDTH, 3])

    return model.predict_classes(image_tensor)[0]


def write_statistics(account_path, statistics):
    statistics_percentage = map(lambda x: 100 * x / sum(statistics), statistics)
    with open(os.path.join(account_path, configs.INFO_FILENAME), 'w+') as f:
        f.write(f'{configs.VK_URL}/id{os.path.basename(account_path)}\n')
        for name, statistic in zip(configs.CATEGORIZER_CLASSES, statistics_percentage):
            f.write(f'{name} - {statistic:.3f}%\n')


def predict(logger, max_girls_count, binary_model_path, categorizer_model_path):
    girls_count = 0

    binary_filter = tf.keras.models.load_model(binary_model_path)
    categorizer = tf.keras.models.load_model(categorizer_model_path)

    accounts_paths = utils.get_paths(configs.ACCOUNTS_PATH)
    logger.info(f'Found {len(accounts_paths)} girls in {configs.ACCOUNTS_PATH}')

    for account_path in accounts_paths:
        faces = find_faces(logger, account_path)

        lit_counter = 0
        predictions_statistics = [0] * len(configs.CATEGORIZER_CLASSES)
        
        for face in faces:
            if configs.BINARY_FILTER_CLASSES[get_prediction(binary_filter, face[0])[0]] == 'LIT':
                logger.warning(f'Face \'{face[1]}\' is lit')
                lit_counter += 1
                continue
            
            predictions_statistics[get_prediction(categorizer, face[0])] += 1

        if lit_counter == len(faces):
            utils.move_folder(logger, account_path, configs.FAILED_PATH)
        else:
            logger.info(f'Prediction statistic: {predictions_statistics}')
            write_statistics(account_path, predictions_statistics)

            intensified_prediction = predictions_statistics.index(max(predictions_statistics))
            class_path = os.path.join(configs.SUCCESS_PATH, configs.CATEGORIZER_CLASSES[intensified_prediction])
            utils.move_folder(logger, account_path, class_path)

        girls_count += 1
        if girls_count == max_girls_count:
            break

    logger.info(f'{girls_count} girl(s) was predicted')


def main():
    args = parse_args()

    app_name = os.path.splitext(os.path.basename(__file__))[0]
    logger = utils.get_logger(app_name)

    predict(logger, args.max_girls, args.binary_filter_model, args.categorizer_model)


if __name__ == '__main__':
    main()
