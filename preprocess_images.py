from curses import raw
from operator import gt
import os
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
from utils import read_files_from_path, read_pairs, read_image, read_image_16bit
from generate_pairs import generate_valid_poses, generate_valid_image_dictionary
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


def canny_edge(cv_img, show=True):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_img = clahe.apply(cv_img)
    # cv_img = cv2.equalizeHist(cv_img)
    edges = cv2.Canny(eq_img, 100, 200)

    added_image = cv2.addWeighted(cv_img,0.8, edges,0.2,0)
    im_v = cv2.vconcat([added_image, cv_img])
    cv2.imshow("added_image", im_v)
    cv2.waitKey(0)


def calculate_histogram(cv_img, h, w, size=30):
    # h, w = cv_img.shape
    # # print(w, h)
    # n_pixel = cv_img.size

    hist = np.zeros(size)
    np_img = np.asarray(cv_img, dtype="uint16")
    min, max = np_img.min(), np_img.max()
    interval = (max - min) / size
    vector = np.squeeze(np.floor((np_img - min) / (interval + 0.000001)).reshape(1, -1))
    print(vector.shape, vector.min(), vector.max(), interval) # (327680,) 0.0 29.0 101.76666666666667
    for i in vector:
        i = i.astype(int)
        # print(i, i.dtype, i.shape)
        hist[i] += 1    
    # print(hist, min, max)        

    alpha = hist / vector.shape[0]
    beta = np.array([alpha[:i].sum() for i in range(alpha.shape[0])]) # beta[0] = 0, beta[1] = alpha[0], ...
    print(alpha, beta, alpha.sum())

    idx_img = np.floor((np_img - min) / (interval+ 0.000001))
    print(idx_img.shape, idx_img.dtype)
    idx_img = idx_img.astype(np.uint8)
    print(idx_img, idx_img.max())

    alpha_mat = np.zeros([h, w])
    beta_mat = np.zeros([h, w])
    for y in range(0, h):
        for x in range(0, w):
            # print(y, x, idx_img[y,x].astype(int))
            idx = idx_img[y,x].astype(int)
            alpha_mat[y,x] = alpha[idx]
            beta_mat[y,x] = alpha[:idx].sum()

    assert (x - idx_img*interval).all() >= 0
    new_img = alpha_mat * (x - idx_img*interval) / interval + beta_mat
    print(new_img.min(), new_img.max())
    hist_np, bin_edges = np.histogram(new_img, density=True)
    # plt.hist(hist_np, bins=100, density=True)
    
    imshow_img = (new_img * 255).astype(np.uint8)
    # plt.imshow(imshow_img, cmap='gray')
    # plt.show()
    return imshow_img


    # (np_img - bi_img) / interval

'''
data-driven min, max calculation
'''
def calculate_minmax_from_histogram(src_img_folder, sample_size=100, 
                low_percentage=0.15, high_percentage=0.03, savefilename=None):
    img_list = sorted(os.listdir(src_img_folder), reverse=False)
    # print(img_list[:10])
    first_img = cv2.imread(os.path.join(src_img_folder,img_list[0]), -1)
    hist = cv2.calcHist([first_img], [0], None, [65535], [0,65535]) / sample_size
    for i in np.random.choice(len(img_list), sample_size-1):
        raw_img = cv2.imread(os.path.join(src_img_folder,img_list[i]), -1)
        crop_img = crop_image(raw_img)
        hist += cv2.calcHist([crop_img], [0], None, [65535], [0,65535]) / sample_size
    
    plt.cla()
    plt.plot(hist, color='b')
    plt.xlim(2000, 5000)

    if savefilename is not None:
        plt.savefig(savefilename)

    np_hist = np.asarray(hist)
    np_hist.sort()
    sum = np_hist.sum()
    
    min = -1
    max = -1
    forward_total = 0
    for idx in range(np_hist.shape[0]):
        forward_total += np_hist[idx]
        if forward_total > low_percentage * sum:
            min = idx
            break
    backward_total = 0
    for idx in range(np_hist.shape[0])[::-1]:
        backward_total += np_hist[idx]
        if backward_total > high_percentage * sum:
            max = idx
            break
    
    return min, max


def calculate_minmax_heuristic(seq):

    if seq == "01":
        min, max = 2900, 3800
    elif seq == "02":
        min, max = 3050, 4200 # chosen
    elif seq == "03":
        min, max = 3000, 3300 # probable
    elif seq == "04":
        min, max = 2900, 3800 # probable
    elif seq == "05":
        min, max = 2900, 4600
    elif seq == "06":
        min, max = 2950, 3650
    elif seq == "07":
        min, max = 3000, 4500
    elif seq == "08":
        min, max = 3200, 3750
    elif seq == "09":
        min, max = 2900, 3550
    else:
        min, max = 2000, 4000

    return min, max


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


# not working
def plot_histogram_from_image_sequences(src_img_folder, dst_img_folder, gt_pose_path, K, D, h, w, raw=False):

    valid_ids_imgs = generate_valid_image_dictionary(gt_pose_path)
    seq = str(dst_img_folder[-15:-13])
    print("seq: ", seq)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    for idx, img in valid_ids_imgs:
        
        '''
        code for raw 14bit images
        '''
        raw_img = cv2.imread(os.path.join(src_img_folder,img), -1)
        undist_img = cv2.undistort(raw_img, K, D, None)
        raw_img_crop = crop_image(undist_img, deepcopy=True)
        hist = cv2.calcHist([raw_img_crop], [0], None, [65535], [0,65535])

        '''
        debug: min-max equalize 14 bit images
        '''    
        min, max = 0, 4096
        # undist_img = cv2.undistort(raw_img, K, D, None)
        crop_img = crop_image(undist_img, h, w, deepcopy=False)
        for y in range(0, 400):
            for x in range(0, 640):
                if crop_img[y,x] >= max:
                    crop_img[y,x] = max
                elif crop_img[y,x] <= min:
                    crop_img[y,x] = min
                crop_img[y,x] = np.floor(min + max * (crop_img[y,x] - min) / (max - min))
        final_img = crop_img.astype('uint16')
        hist_equalized = cv2.calcHist([final_img], [0], None, [65535], [0,65535])

        '''
        code for normalized 8bit image
        '''
        # raw_img_8bit = cv2.imread(os.path.join(dst_img_folder,to6digit(idx)+'.png'), 0)
        # crop_img = crop_image(raw_img_8bit, copy=False)
        # hist = cv2.calcHist([crop_img], [0], None, [256], [0,256])


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
        
        ax1.imshow(raw_img_crop ,cmap=plt.get_cmap('gray'))
        ax2.imshow(final_img ,cmap=plt.get_cmap('gray'), vmin=3000, vmax=3500)
        
        ax3.plot(hist, color='b')
        ax3.set_xlim([min, max])
        ax3.set_ylim([0, 2500])
        
        ax4.plot(hist_equalized, color='b')
        ax4.set_xlim([min, max])
        ax4.set_ylim([0, 2500])
        
        plt.draw()
        plt.pause(0.01)


def read_and_save_cropped_images(src_img_folder, dst_img_folder, K, D, h, w):
    seq = str(dst_img_folder[-15:-13])
    print("seq: ", seq)
    print("calculating min, max...")
    # min, max = calculate_minmax_from_histogram(src_img_folder, 
    #                     savefilename='/media/hj/seagate/datasets/STHEREO/histogram/'+seq+'.jpg')
    # print("intensity min, max: ", min, max)
    min, max = calculate_minmax_heuristic(dst_img_folder[-15:-13])
    print("intensity min, max: ", min, max)

    img_list = sorted(os.listdir(src_img_folder), reverse=False)
    count = 0
    # print(img_list[:10])
    for img in img_list:
        print(img)
        # 16 bit
        raw_img = cv2.imread(os.path.join(src_img_folder,img), -1)
        undist_img = cv2.undistort(raw_img, K, D, None)
        crop_img = crop_image(undist_img, h, w)
        # cv2.normalize(crop_img, crop_img, 0, 65535, cv2.NORM_MINMAX)
        # plt.imshow(calculate_histogram(crop_image), cmap='gray')
        # plt.show()
        # 8 bit
        # raw_img = cv2.imread(os.path.join(src_img_folder,img), 0)
        # undist_img = cv2.undistort(raw_img, K, D, None)
        # crop_img = crop_image(undist_img, h, w)
        # cv2.normalize(crop_img, crop_img, 0, 255, cv2.NORM_MINMAX)
        # crop_img = cv2.applyColorMap(crop_img, cv2.COLORMAP_JET)

        for y in range(0, h):
            for x in range(0, w):
                if crop_img[y,x] >= max:
                    crop_img[y,x] = max
                elif crop_img[y,x] <= min:
                    crop_img[y,x] = min
                crop_img[y,x] = 255 * (crop_img[y,x] - min) / (max - min)

        crop_img = crop_img.astype('uint8')



        cv2.imshow("crop_img", crop_img)
        cv2.waitKey(1)
        # crop_img = cv2.applyColorMap(crop_img, cv2.COLORMAP_JET)
        cv2.imwrite((os.path.join(dst_img_folder,img)), crop_img)
        
        count += 1
        # if count == 1:
        #     break
        if count % 10 == 0:
            print('total count: ', count)



def visualize_two_images_pair(img1_path, img2_path, img1_seq, img2_seq, loop):

    img1 = np.array(cv2.imread(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = np.array(cv2.imread(img2_path), cv2.IMREAD_GRAYSCALE)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    img1 = cv2.putText(img1, 'id=' + str(img1_seq), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    img2 = cv2.putText(img2, 'id=' + str(img2_seq), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    img_pair = np.concatenate((img1, img2), axis=1)
    img_pair = cv2.putText(img_pair, 'loop: '+str(loop), (350,30), font, 
                    fontScale, color, thickness, cv2.Line_AA)

    cv2.imshow('loop', img_pair)
    cv2.waitKey(0) 
    


def visualize_images_from_training_sets(data_root_folder, target_seq):

    img1_root_dir = os.path.join(data_root_folder, target_seq)
    train_pairs = read_pairs(os.path.join(img1_root_dir, 'train_sets/train_pairs.csv'))
    img1_idx_table = read_pairs(os.path.join(img1_root_dir, 'train_sets/global_pose.csv'))
    
    img2_seq = -1
    img2_root_dir = ''
    img2_idx_table = np.zeros(0)
    

    for idx in range(train_pairs.shape[0]):

        img1_index = train_pairs[idx, 0]
        img2_index = train_pairs[idx, 1]
        dir1_index = train_pairs[idx, 2]
        dir2_index = train_pairs[idx, 3]
        is_loop = train_pairs[idx, 4]

        # load image idx tables
        if img2_seq == -1 or past_dir2_index != dir2_index:
            img2_root_dir = os.path.join(data_root_folder, dir2_index)
            img2_idx_table = read_pairs(os.path.join(img2_root_dir, 'train_sets/global_pose.csv'))

        img1_path = os.path.join(img1_root_dir, 'thermal_left', img1_idx_table(img1_index-1))
        img2_path = os.path.join(img2_root_dir, 'thermal_left', img2_idx_table(img2_index-1))

        visualize_two_images_pair(img1_path, img2_path, dir1_index, dir2_index, is_loop)
        past_dir2_index = dir2_index


def preprocess_image(cv_img, K, D, h, w, min, max):
    undist_img = cv2.undistort(cv_img, K, D, None)
    crop_img = crop_image(undist_img, h, w, deepcopy=False)
    
    # for y in range(0, h):
    #     for x in range(0, w):
    #         if crop_img[y,x] >= max:
    #             crop_img[y,x] = max
    #         elif crop_img[y,x] <= min:
    #             crop_img[y,x] = min
    #         crop_img[y,x] = np.floor(255 * (crop_img[y,x] - min) / (max - min))
    # final_img = crop_img.astype('uint8')
    final_img = crop_img.astype('uint16')
    
    return final_img


def preprocess_and_save_valid_image(src_img_folder, dst_img_folder, gt_pose_path, K, D, h, w):
    
    valid_ids_imgs = generate_valid_image_dictionary(gt_pose_path)

    seq = str(dst_img_folder[-15:-13])
    print("seq: ", seq)
    min, max = calculate_minmax_heuristic(dst_img_folder[-15:-13])
    # min, max = 2800, 3350
    print("intensity min, max: ", min, max)

    for idx, img in valid_ids_imgs:
        raw_img = cv2.imread(os.path.join(src_img_folder,img), -1)
        out_img = preprocess_image(raw_img, K, D, h, w, min, max)

        # min, max = 2700, 3350
        # out_img1 = preprocess_image(raw_img, K, D, h, w, min, max)
        # min, max = 2900, 3550
        # out_img2 = preprocess_image(raw_img, K, D, h, w, min, max)
        # add_img = cv2.addWeighted(out_img1, 0.5, out_img2, 0.5, 0.0)
        # out_img = cv2.vconcat([out_img1, out_img2, add_img])

        # hist = cv2.calcHist([out_img], [0], None, [256], [0, 255])
        # cv2.imshow("hist", hist); cv2.waitKey(0); break
        # hist /= hist.sum()
        # plt.cla()
        # plt.plot(hist) 
        # plt.show() ; break

        # print((os.path.join(dst_img_folder,to6digit(idx)+'.png')))
        # break

        cv2.imshow(seq, out_img.astype('uint8')); cv2.waitKey(1)
        cv2.imwrite((os.path.join(dst_img_folder,to6digit(idx)+'.png')), out_img)


if __name__ == '__main__':
    
    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))

    dst_root_path = config["dst_root_path"]
    interpolated_gt_path = config["interpolated_gt_path"]

    raw_image_folder = config["raw_image_folder"]
    preprocessed_image_folder = config["preprocessed_image_folder"]
    camera_matrix = np.array(config["camera_matrix"]["data"], dtype=np.float64).reshape(3,3)
    distort_coeffs = np.array(config["distortion_coefficients"]["data"], dtype=np.float64).reshape(1,5)
    seq = config["seq"]

    for seq in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
    # for seq in ["01", "02", "03"]:
    # for seq in ["04", "05", "06"]:
    # for seq in ["07", "08", "09"]:
        src_img_folder = os.path.join(raw_image_folder, seq, "image/stereo_thermal_14_left")
        dst_img_folder = os.path.join(preprocessed_image_folder, seq, "thermal_left")
        img_list = sorted(os.listdir(src_img_folder), reverse=False)
        
        # read_and_save_cropped_images(src_img_folder, dst_img_folder, camera_matrix, distort_coeffs, 400, 640)
        
        preprocess_and_save_valid_image(src_img_folder, dst_img_folder, 
                os.path.join(dst_root_path, seq, interpolated_gt_path), camera_matrix, distort_coeffs, 400, 640)
        
        # plot_histogram_from_image_sequences(src_img_folder, dst_img_folder, 
        #         os.path.join(dst_root_path, seq, interpolated_gt_path), camera_matrix, distort_coeffs, 400, 640)
        
        # for img in img_list:
        #     normalized_img = read_image(os.path.join(src_img_folder, img))
        #     cv2.imshow('test', normalized_img.astype('uint8'))
        #     cv2.waitKey(1)
        
        # break