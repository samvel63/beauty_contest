import sys

import cv2

from keras.applications import ResNet50


def main():
    image_path = sys.argv[1]

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print("[INFO] Found {0} Faces!".format(len(faces)))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # status = cv2.imwrite('faces_detected.jpg', image)
    # print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    for ind, face in enumerate(faces):
        x, y, w, h = [v for v in face]
        face_image = image[y:y+h, x:x+w]

        resized = cv2.resize(face_image, (300, 300), interpolation=cv2.INTER_AREA)
        status = cv2.imwrite(f'faces/face_{ind}.jpg', resized)

        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


if __name__ == '__main__':
    main()
