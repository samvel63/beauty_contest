import os
import sys
import logging

import cv2


def get_paths(path):
    return sorted(os.path.join(path, file) for file in os.listdir(path))


def save_faces(image_path, output_path):
    global sum_images

    image_name = os.path.basename(image_path)
    logger.warning(f'Read {image_name}')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width, channels = image.shape

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(80, 80)
    )
    logger.warning(f'Found {len(faces)} faces for {image_name}!')

    for ind, face in enumerate(faces):
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
        face_image_name = f'face_{sum_images}.jpg'
        face_image_path = os.path.join(output_path, face_image_name)
        resized = cv2.resize(face_image, (150, 150), interpolation=cv2.INTER_AREA)

        cv2.imwrite(face_image_path, resized)
        sum_images += 1

        logger.warning(f'{image_name} => {face_image_name} have written to {output_path}')


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    logger.warning(f'Searching for faces in {input_path} and saving to {output_path}')

    for file_path in get_paths(input_path):
        try:
            save_faces(file_path, output_path)
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    sum_images = 1
    app_name = os.path.splitext(os.path.basename(__file__))[0]

    logger = logging.getLogger(app_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)

    if len(sys.argv) != 3:
        logger.error(f'Bad input parameters: `{app_name}.py input_path output_path`')
        exit(0)

    main()
