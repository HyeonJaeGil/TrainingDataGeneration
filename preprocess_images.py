from operator import gt
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import yaml
import math
from utils import read_files_from_path
np.set_printoptions(precision=9)


def crop_image(img, h, w):
    h_orig, w_orig, _ = img.shape
    assert h_orig >= h
    assert w_orig >= w

    crop_img = img[0:h, 0:w]
    return crop_img

# undistortion needed (TODO)
def read_and_save_cropped_images(src_img_folder, dst_img_folder, h, w):

    count = 0
    img_list = sorted(os.listdir(src_img_folder), reverse=False)
    print(img_list[:10])
    for img in img_list:
        raw_img = cv2.imread(os.path.join(src_img_folder,img))
        crop_img = crop_image(raw_img, h, w)
        cv2.normalize(crop_img, crop_img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite((os.path.join(dst_img_folder,img)), crop_img)
        
        count += 1
        if count % 100 == 0:
            print('total count: ', count)
            # break


if __name__ == '__main__':
    
    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))

    src_img_folder = config["raw_image_folder"]
    dst_img_folder = config["preprocessed_image_folder"]
    img_list = sorted(os.listdir(src_img_folder), reverse=False)

    read_and_save_cropped_images(src_img_folder, dst_img_folder, 400, 640)