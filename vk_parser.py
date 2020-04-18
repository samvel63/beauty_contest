import os
import sys
import logging
import requests

import vk_api as vk


app_name = os.path.splitext(os.path.basename(__file__))[0]
root_path = os.path.dirname(os.path.abspath(__file__))
accounts_path = os.path.join(root_path, 'ACCOUNTS')
success_path = os.path.join(root_path, 'SUCCESS')
failed_path = os.path.join(root_path, 'FAILED')

VK_URL = 'https://vk.com'
LOGIN = '+79958812067'
PASSWORD = 'peshuesos228'

GIRLS_PER_REQ = 5  # MAX 1000
PHOTOS_PER_REQ = 200  # MAX 200
MAX_PHOTOS = 50
CLASSES = ['asian', 'brunette', 'dreadlock', 'ginger', 'mulatto']


def get_vk_api():
    logger.info('Getting VK API')

    vk_session = vk.VkApi(LOGIN, PASSWORD)
    vk_session.auth()

    return vk_session.get_api()


def get_girls(vk_api, count, offset):
    # sort: 0 - sort by popularity
    # city: 1 - Moscow
    # sex: 1 - female
    # has_photo: 1 - has photo on a page
    return vk_api.users.search(sort=0, offset=offset, count=count, city=1, sex=1, age_from=17, age_to=40, has_photo=1)


def exists_girl(girl_id):
    is_exists = False

    paths_to_check = list()

    paths_to_check.append(os.path.join(accounts_path, girl_id))
    paths_to_check.append(os.path.join(failed_path, girl_id))

    for girl_type in CLASSES:
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


def download_girl_photos(vk_api, girl_id, max_photos_count):
    photos_count = 0

    girl_url = f'{VK_URL}/id{girl_id}'
    logger.info(f'Searching photos on the page {girl_url}')

    photos = get_photos(vk_api, girl_id, PHOTOS_PER_REQ, 0)
    logger.info(f'Found {photos["count"]} photos on the page {girl_url}')

    for offset in range(0, photos['count'], PHOTOS_PER_REQ):
        logger.info(f'Get photos from page {girl_url} in range [{offset}, {offset+PHOTOS_PER_REQ})')
        photos = get_photos(vk_api, girl_id, PHOTOS_PER_REQ, offset)

        for photo in photos['items']:
            for photo_size in photo['sizes']:
                if photo_size['type'] == 'z':
                    photo_name = f'{photos_count}.jpg'
                    photo_path = os.path.join(accounts_path, girl_id, photo_name)

                    download_photo(photo_size['url'], photo_path)
                    photos_count += 1

                    if photos_count >= max_photos_count:
                        return photos_count

    return photos_count


def download_photo(photo_url, photo_path):
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


def main():
    vk_api = get_vk_api()

    if not os.path.exists(accounts_path):
        os.makedirs(accounts_path)

    girls = get_girls(vk_api, GIRLS_PER_REQ, 0)
    logger.info(f'Founded {girls["count"]} girls')

    # for offset in range(0, girls["count"], GIRLS_PER_REQ):
    for offset in range(0, 1, GIRLS_PER_REQ):
        logger.info(f'Get girls in range [{offset}, {offset+GIRLS_PER_REQ})')
        girls = get_girls(vk_api, GIRLS_PER_REQ, offset)

        for girl in girls['items']:
            girl_id = str(girl['id'])

            if not exists_girl(girl_id):
                girl_url = f'{VK_URL}/id{girl_id}'
                girls_account_path = os.path.join(accounts_path, girl_id)

                os.makedirs(girls_account_path, exist_ok=True)
                photos_count = download_girl_photos(vk_api, girl_id, MAX_PHOTOS)
                logger.info(f'Downloaded {photos_count} photos from page {girl_url} to {girls_account_path}')


if __name__ == '__main__': 
    logger = logging.getLogger(app_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    main()
