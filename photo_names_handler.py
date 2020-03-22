import argparse
import random
import string
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', help='Name of photos `{name}.1.jpg ... {name}.N.jpg`', required=True)
    parser.add_argument('--path', '-p', help='Path of photos', required=True)

    return parser.parse_args()


def random_string(string_length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(string_length))


def main():
    args = parse_args()

    photos_list = os.listdir(args.path)
    random_name = random_string()
    for ind, photo_name in enumerate(photos_list):
        new_name = f'{random_name}.{ind+1}.jpg'

        before = os.path.join(args.path, photo_name)
        after = os.path.join(args.path, new_name)

        os.rename(before, after)

    photos_list = os.listdir(args.path)
    for ind, photo_name in enumerate(photos_list):
        new_name = f'{args.name}.{ind+1}.jpg'

        before = os.path.join(args.path, photo_name)
        after = os.path.join(args.path, new_name)

        os.rename(before, after)


if __name__ == '__main__':
    main()
