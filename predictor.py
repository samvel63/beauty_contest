import os
import sys
import logging

import cv2


root_path = os.path.dirname(os.path.abspath(__file__))
accounts_path = os.path.join(root_path, 'ACCOUNTS')
success_path = os.path.join(root_path, 'SUCCESS')
failed_path = os.path.join(root_path, 'FAILED')

image_size_width = 150
image_size_height = 150



def get_paths(path):
    return sorted(os.path.join(path, file) for file in os.listdir(path))






def find_faces(account_path):
    
    def get_resized_face(face):
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

    found_faces = []
    for file_path in get_paths(account_path):
        image_name = os.path.basename(image_path)
        logger.info(f'Read {image_name}')

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(80, 80)
        )
        logger.info(f'Found {len(faces)} faces for {image_name}!')

        for face in faces:
            found_faces.append(get_resized_face(face))
        
    return found_faces
    

def main():
    for account_path in account_paths:
        faces = find_faces(account_path)
        faces_size = len(faces)
        
        lit_counter = 0
        predictions_statistics = [0] * len(classes)
        
        for face in faces:
            if binary_filter_get_prediction(face) == CLASS_LIT:
                lit_counter += 1
                continue
            
            predictions_statistics[categorizer_get_prediction(face)] += 1

        if lit_counter == faces_size:
            move_to_failed(account_path)
        else:
            intensified_prediction = predictions_statistics.index(max(predictions_statistics))
            move_to_success(account_path, intensified_prediction)


        

            

                







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
        face_image_resized = cv2.resize(face_image, (image_size_width, image_size_height), interpolation=cv2.INTER_AREA)

        


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
