import os.path


VK_URL = 'https://vk.com'
GIRLS_PER_REQ = 1000  # MAX 1000
PHOTOS_PER_REQ = 200  # MAX 200


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ACCOUNTS_PATH = os.path.join(ROOT_PATH, 'ACCOUNTS')
SUCCESS_PATH = os.path.join(ROOT_PATH, 'SUCCESS')
FAILED_PATH = os.path.join(ROOT_PATH, 'FAILED')
INFO_FILENAME = '_INFO'


IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_MIN_HEIGHT_FIND = 80
IMG_MIN_WIDTH_FIND = 80


BINARY_FILTER_CLASSES = ['GIRL', 'LIT']
CATEGORIZER_CLASSES = ['ASIAN', 'BRUNETTE', 'DREADLOCK', 'GINGER', 'MULATTO']
