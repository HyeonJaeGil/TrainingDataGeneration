from curses import raw
from operator import gt
import os
from os.path import join
import sys
from cv2 import DIST_FAIR
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
import copy
import time
from utils import read_image, read_image_16bit
from generate_pairs import generate_valid_poses, get_valid_image_dict
np.set_printoptions(precision=9)


def to2digit(number):
    if type(number) == float:
        number = int(number)
    count = str(number)
    return ('0' * (2-len(count)) + count)


def to6digit(number):
    if type(number) == float:
        number = int(number)
    count = str(number)
    return ('0' * (6-len(count)) + count)


def crop_image(img, h=400, w=640, deepcopy=True):
    h_orig, w_orig = img.shape
    assert h_orig >= h
    assert w_orig >= w

    if deepcopy:
        img_clone = copy.deepcopy(img)
        crop_img = img_clone[0:h, 0:w]
    else:
        crop_img = img[0:h, 0:w]
        
    return crop_img


def plot_histogram_from_image_sequences(src_img_folder, dst_img_folder, gt_pose_path, K, D, h, w, raw=True):

    valid_ids_imgs = get_valid_image_dict(gt_pose_path)
    seq = str(dst_img_folder[-15:-13])
    print("seq: ", seq)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    for idx, img in valid_ids_imgs:
        
        min, max = 0, 4096

        if raw: # 14bit
            raw_img = cv2.imread(join(src_img_folder,img), -1)
            undist_img = cv2.undistort(raw_img, K, D, None)
            undist_img_crop = crop_image(undist_img, deepcopy=True)
            hist = cv2.calcHist([undist_img_crop], [0], None, [65535], [0,65535])

            crop_img = crop_image(undist_img, h, w, deepcopy=False)
            norm_img = 65535 * (crop_img - min) / (max - min) 
            np.clip(norm_img, 0, 65535, out=norm_img)
            final_img = norm_img.astype('uint16')
            hist_equalized = cv2.calcHist([final_img], [0], None, [65535], [0,65535])

        else: # 8bit
            raw_img_8bit = cv2.imread(join(dst_img_folder,to6digit(idx)+'.png'), 0)
            undist_img = cv2.undistort(raw_img_8bit, K, D, None)
            undist_img_crop = crop_image(undist_img, deepcopy=True)            
            hist = cv2.calcHist([undist_img_crop], [0], None, [256], [0,256])

        plt.cla()
        plt.title(seq)

        ax1 = fig1.gca()
        ax2 = fig2.gca()
        ax3 = fig3.gca()
        ax4 = fig4.gca()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        ax1.imshow(undist_img_crop ,cmap=plt.get_cmap('gray'))
        ax2.imshow(final_img ,cmap=plt.get_cmap('gray'), vmin=3000, vmax=3500)
        
        ax3.plot(hist, color='b')
        ax3.set_xlim([min, max])
        ax3.set_ylim([0, 2500])
        
        ax4.plot(hist_equalized, color='b')
        ax4.set_xlim([min, max])
        ax4.set_ylim([0, 2500])
        
        plt.draw()
        plt.pause(0.01)


def preprocess_image(cv_img, K, D, h, w):
    undist_img = cv2.undistort(cv_img, K, D, None)
    crop_img = crop_image(undist_img, h, w, deepcopy=False)
    final_img = crop_img.astype('uint16')
    return final_img


def preprocess_and_save_valid_image(src_img_folder, dst_img_folder, gt_pose_path, K, D, h, w):
    
    seq = str(dst_img_folder[-15:-13])
    print("seq: ", seq)
    valid_ids_imgs = get_valid_image_dict(gt_pose_path)

    for idx, img in valid_ids_imgs:
        raw_img = cv2.imread(os.path.join(src_img_folder,img), -1)
        out_img = preprocess_image(raw_img, K, D, h, w)
        cv2.imshow(seq, out_img.astype('uint8')); cv2.waitKey(1)
        # cv2.imwrite((os.path.join(dst_img_folder,to6digit(idx)+'.png')), out_img)


if __name__ == '__main__':
    
    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))

    src_root_path = config["src_root_path"]
    src_image_folder = config["src_image_folder"]
    dst_root_path = config["dst_root_path"]
    interpolated_gt_path = config["interpolated_gt_path"]

    camera_matrix = np.array(config["camera_matrix"]["data"], dtype=np.float64).reshape(3,3)
    distort_coeffs = np.array(config["distortion_coefficients"]["data"], dtype=np.float64).reshape(1,5)
    height = config["height"]
    width = config["width"]
    
    seq = config["seq"]
    seqs = config["all_seqs"]

    # for seq in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
    # for seq in ["01", "02", "03"]:
    # for seq in ["04", "05", "06"]:
    # for seq in ["07", "08", "09"]:
    for seq in ["10"]:                
        preprocess_and_save_valid_image(
                    join(src_root_path, seq, src_image_folder),
                    join(dst_root_path, seq, "thermal_left"),
                    join(dst_root_path, seq, interpolated_gt_path), 
                    camera_matrix, distort_coeffs, height, width)
        
        # plot_histogram_from_image_sequences(
        #             join(src_root_path, seq, src_image_folder),
        #             join(dst_root_path, seq, "thermal_left"),
        #             join(dst_root_path, seq, interpolated_gt_path), 
        #             camera_matrix, distort_coeffs, height, width)
