import sys
import os


dir_path = sys.argv[1]
photos_list = os.listdir(dir_path)

k = 0
for photo_name in photos_list:
    k += 1
    before = dir_path + photo_name
    after = dir_path + "mulatto."+str(k) + ".jpg"
    os.rename(before, after)
