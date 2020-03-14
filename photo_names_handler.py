
import os


dir_path = "/home/dmbobyl1/Desktop/BeautyClassification/data_validate/asian/"


photos_list = os.listdir(dir_path)

k = 0
for photo_name in photos_list:
    k += 1
    before = dir_path + photo_name
    after  = dir_path + "asian."+str(k) + ".jpg"
    os.rename(before, after)
