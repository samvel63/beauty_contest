import os
import sys
import logging
import argparse
import requests

import vk_api as vk
from vk_api.exceptions import ApiError

root_path = os.path.dirname(os.path.abspath(__file__))
accounts_path = os.path.join(root_path, 'ACCOUNTS')
success_path = os.path.join(root_path, 'SUCCESS')
failed_path = os.path.join(root_path, 'FAILED')

VK_URL = 'https://vk.com'
GIRLS_PER_REQ = 1000  # MAX 1000
PHOTOS_PER_REQ = 200  # MAX 200
GIRL_TYPES = ['asian', 'brunette', 'dreadlock', 'ginger', 'mulatto']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-girls', '-mg', help='Number of girls per one execute of the program', type=int, required=True)
    parser.add_argument('--max-photos', '-mp', help='Number of photos per one girl', type=int, required=True)

    parser.add_argument('--login', '-l', help='Login of the VK page', required=True)
    parser.add_argument('--password', '-p', help='Password of the VK page', required=True)

    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_vk_api(login, password):
    vk_session = vk.VkApi(login, password)
    vk_session.auth()

    return vk_session.get_api()


def get_girls(vk_api, count, offset):
    # sort: 0 - sort by popularity
    # city: 1 - Moscow
    # sex: 1 - female
    # has_photo: 1 - has photo on a page
    return vk_api.users.search(sort=0, offset=offset, count=count, city=1, sex=1, age_from=17, age_to=40, has_photo=1)


def exists_girl(logger, girl_id):
    is_exists = False

    paths_to_check = list()
    paths_to_check.append(os.path.join(accounts_path, girl_id))  # check in accounts path
    paths_to_check.append(os.path.join(failed_path, girl_id))  # check in failed path

    # check in success path
    for girl_type in GIRL_TYPES:
        paths_to_check.append(os.path.join(success_path, girl_type, girl_id))

    for path in paths_to_check:
        if os.path.exists(path):
            if len(os.listdir(path)) > 0:
                logger.info(f'Girl {girl_id} already in {path}')
                is_exists = True
            else:
                logger.warning(f'Girl {girl_id} already in {path} but without photos!')

    return is_exists


def get_photos(vk_api, girl_id, count, offset):
    return vk_api.photos.getAll(owner_id=girl_id, count=count, offset=offset)


def download_girl_photos(vk_api, logger, girl_id, max_photos_count):
    photos_downloaded = 0

    girl_url = f'{VK_URL}/id{girl_id}'
    logger.info(f'Getting photos from page {girl_url}')

    try:
        photos = get_photos(vk_api, girl_id, PHOTOS_PER_REQ, 0)
    except ApiError as e:
        logger.error(f'Failed to get photos from page {girl_url} with message: {e}')
        return 0

    photos_count = photos.get('count')
    logger.info(f'Found {photos_count} photos on the page {girl_url}')

    for offset in range(0, photos_count, PHOTOS_PER_REQ):
        logger.info(f'Get photos from page {girl_url} in range [{offset}, {offset+PHOTOS_PER_REQ})')
        photos = get_photos(vk_api, girl_id, PHOTOS_PER_REQ, offset)

        for photo in photos['items']:
            for photo_size in photo['sizes']:
                if photo_size['type'] == 'z':
                    photo_name = f'{photos_downloaded}.jpg'
                    photo_path = os.path.join(accounts_path, girl_id, photo_name)

                    download_photo(logger, photo_size['url'], photo_path)
                    photos_downloaded += 1

                    if photos_downloaded == max_photos_count:
                        return photos_downloaded

    return photos_downloaded


def download_photo(logger, photo_url, photo_path):
    logger.info(f'Downloading photo {photo_url} to {photo_path}')
    with open(photo_path, 'wb') as photo:
        response = requests.get(photo_url, stream=True)
        if not response.ok:
            logger.error(response)
            sys.exit(0)

        for block in response.iter_content(1024):
            if not block:
                break
            photo.write(block)


def load_girls(vk_api, logger, max_girls_count, max_photos_count):
    girls_loaded = 0
    if not os.path.exists(accounts_path):
        os.makedirs(accounts_path)

    girls = get_girls(vk_api, GIRLS_PER_REQ, 0)
    girls_count = girls.get('count')
    logger.info(f'Founded {girls_count} girls')

    for offset in range(0, girls_count, GIRLS_PER_REQ):
        logger.info(f'Get girls in range [{offset}, {offset+GIRLS_PER_REQ})')
        girls = get_girls(vk_api, GIRLS_PER_REQ, offset)

        for girl in girls['items']:
            girl_id = str(girl['id'])

            if not exists_girl(logger, girl_id):
                girls_account_path = os.path.join(accounts_path, girl_id)
                os.makedirs(girls_account_path, exist_ok=True)

                photos_count = download_girl_photos(vk_api, logger, girl_id, max_photos_count)
                logger.warning(f'Downloaded {photos_count} photos for the girl {girl_id} to {girls_account_path}')

                girls_loaded += 1
                if girls_loaded == max_girls_count:
                    return girls_loaded

    return girls_loaded


def main():
    args = parse_args()

    app_name = os.path.splitext(os.path.basename(__file__))[0]
    logger = get_logger(app_name)

    logger.info(f'Getting VK API for login={args.login}, password={args.password}')
    vk_api = get_vk_api(args.login, args.password)

    girls_count = load_girls(vk_api, logger, args.max_girls, args.max_photos)
    logger.info(f'Loaded {girls_count} girls')


if __name__ == '__main__':
    main()
