import os
import sys
import logging

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS_PATH = os.path.join(ROOT_PATH, 'ACCOUNTS')
SUCCESS_PATH = os.path.join(ROOT_PATH, 'SUCCESS')
FAILED_PATH = os.path.join(ROOT_PATH, 'FAILED')



IMG_HEIGHT = 150
IMG_WIDTH = 150

IMG_MIN_HEIGHT_FIND = 80
IMG_MIN_WIDTH_FIND = 80



BINARY_FILTER_CLASSES = ['GIRL', 'LIT']
CATEGORIZER_CLASSES = ['ASIAN', 'BRUNETTE', 'DREADLOCK', 'GINGER', 'MULATTO']



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