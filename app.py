import os
import sys

import cv2


sum_images = 0


def get_paths(path):
    return [os.path.join(path, file) for file in os.listdir(path)]


def save_faces(image_path):
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
        minSize=(30, 30)
    )
    print(f'[INFO] Found {len(faces)} Faces for {image_path} image!')
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # status = cv2.imwrite('faces_detected.jpg', image)
    # print('[INFO] Image faces_detected.jpg written to filesystem: ', status)

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
        face_image_name = f'face_{sum_images}_{ind}.1.jpg'

        resized = cv2.resize(face_image, (75, 75), interpolation=cv2.INTER_AREA)
        status = cv2.imwrite(f'faces/{face_image_name}', resized)

        print(f'[INFO] Image {face_image_name} written to filesystem: ', status)

    if len(faces) > 0:
        sum_images += 1


if __name__ == '__main__':

    for image_path in get_paths(sys.argv[1]):
        save_faces(image_path)
