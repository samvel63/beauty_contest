import os
import logging
import requests

import vk_api as vk


root_path = os.path.dirname(os.path.abspath(__file__))
app_name = os.path.splitext(os.path.basename(__file__))[0]

accounts_path = os.path.join(root_path, 'accounts')
results_path = os.path.join(root_path, 'results')

VK_URL = 'https://vk.com'
LOGIN = '+79958812067'
PASSWORD = 'peshuesos228'
GIRLS_COUNT = 5  # MAX 1000
ALBUMS_COUNT = 20  # MAX 200
PHOTOS_COUNT = 50


def get_vk_api():
    logger.warning('Getting VK API')

    vk_session = vk.VkApi(LOGIN, PASSWORD)
    vk_session.auth()

    return vk_session.get_api()


def get_girls(vk_api, count):
    logger.warning('Looking for girls')

    # sort: 0 - sort by popularity
    # city: 1 - Moscow
    # sex: 1 - female
    # has_photo: 1 - has photo on a page
    girls = vk_api.users.search(sort=0, count=count, city=1, sex=1,
                                age_from=18, age_to=35, has_photo=1)

    logger.warning(f'Founded {girls["count"]} girls')
    logger.warning(f'Got {len(girls["items"])} girls')

    return girls


def get_photos(vk_api, owner_id, count):
    profile_url = f'{VK_URL}/id{owner_id}'

    logger.warning(f'Searching for photos at {profile_url}')

    photos = vk_api.photos.getAll(owner_id=owner_id, count=count, no_service_albums=1)

    logger.warning(f'Founded {photos["count"]} albums at {profile_url}')
    logger.warning(f'Got {photos["items"]} albums at {profile_url}')

    return photos


def save_image(owner_id, image_name, image_url):
    image_path = os.path.join(accounts_path, owner_id, image_name)

    with open(image_path, 'wb') as handle:
        response = requests.get(image_url, stream=True)
        if not response.ok:
            logger.error(response)

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)


def main():
    vk_api = get_vk_api()

    if not os.path.exists(accounts_path):
        os.makedirs(accounts_path)

    girls = get_girls(vk_api, GIRLS_COUNT)
    print(girls['items'][0])
    for girl in girls:
        girl_id = girl['id']

        girl_account_path = os.path.join(accounts_path, girl_id)
        girl_result_path = os.path.join(results_path, girl_id)

        if os.path.exists(girl_account_path) and len(os.listdir(girl_account_path)) > 0:
            logger.warning(f'Girl {girl_id} already in accounts_path: {accounts_path}')
            continue
        elif os.path.exists(girl_result_path) and len(os.listdir(girl_result_path)) > 0:
            logger.warning(f'Girl {girl_id} already in results_path: {results_path}')
            continue

        os.makedirs(girl_account_path, exist_ok=True)

    photos = get_photos(vk_api, girls['items'][0]['id'], ALBUMS_COUNT)
    print(len(photos))
    print(photos.keys())
    print(photos['count'])
    print(len(photos['items']))

    # count = 0
    # for item in photos['items']:
    #     print(len(item['sizes']))
    #     count += len(item['sizes'])
    # print(count)

    print(photos['items'][0])


if __name__ == '__main__': 
    logger = logging.getLogger(app_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%y/%m/%d %H:%M:%S')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)

    main()
