import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()


def getImagesAndLabels(path):
    # Создаем список файлов в папке patch
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]  # Тут храним масив картинок
    ids = []  # Храним id лица
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        # Переводим изображение в серый, тренер понимает только одноканальное изображение
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces.append(img)
        # Получаем id из названия
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        ids.append(id)
    return faces, ids


if __name__ == '__main__':
    faces, ids = getImagesAndLabels('faces')
    # Тренируем train(данные, id)
    recognizer.train(faces, np.array(ids))
    # Сохраняем результат
    recognizer.write('face.yml')
