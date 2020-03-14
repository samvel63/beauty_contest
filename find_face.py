import os
import sys

import cv2


sum_images = 0


def get_paths(path):
    return [os.path.join(path, file) for file in os.listdir(path)]


def save_faces(image_path, out):
    global sum_images

    print(image_path)

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
    print(f'[INFO] Found {len(faces)} Faces for {image_path} image!')

    for ind, face in enumerate(faces):
        x, y, w, h = [v for v in face]
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

        resized = cv2.resize(face_image, (150, 150), interpolation=cv2.INTER_AREA)
        status = cv2.imwrite(f'{out}/{face_image_name}', resized)

        print(f'[INFO] Image {face_image_name} written to filesystem: ', status)

    if len(faces) > 0:
        sum_images += 1


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Bad input parameters: provide paths to input and output folders')
        exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print(f'[INFO] Searching for faces in {input_path}')

    for file_path in get_paths(input_path):
        save_faces(file_path, output_path)
