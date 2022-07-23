from operator import gt
import os
from os.path import join
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
np.set_printoptions(precision=9)


def to2digit(input):
    number = str(input).split('.')[0]
    assert '.' not in number

    return ('0' * (2-len(number)) + number)


def to6digit(input):
    number = str(input).split('.')[0]
    assert '.' not in number

    return ('0' * (6-len(number)) + number)


# read csv file
def read_csv(file_path): 

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
    
    return np.asarray(rows)
    # return np.asarray(rows, dtype='float64')


# read csv file that contains local (interpolated) pose
def read_interpolated_pose(file_path): 

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append([row[0]] + row[2:])
    
    return np.asarray(rows, dtype='float64')


def read_image(img_path):
    if not os.path.isfile(img_path):
        print(img_path, " [invalid image]")
        return np.array([])
    img = cv2.imread(img_path, -1).astype('float32')
    min, max = np.amin(img), np.amax(img)
    # min, max = 2500, 4000
    img = (img - min) / (max - min)
    np.clip(img, 0, 1, out=img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img = np.expand_dims(img, axis=-1)
    return img


def read_image_16bit(img_path):
    if not os.path.isfile(img_path):
        print(img_path, " [invalid image]")
        return np.array([])
    img = cv2.imread(img_path, -1).astype('float32')
    min, max = np.amin(img), np.amax(img)
    # min, max = 2500, 4000
    img = (img - min) / (max - min)
    np.clip(img, 0, 1, out=img)
    img = np.expand_dims(img, axis=-1)
    return img


def find_nearest_leftmost(array, value):
    
    if np.min(array) > value: 
        return -1
    elif np.max(array) < value: 
        return len(array)

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and value - float(array[idx]) <= 0: 
        return idx-1
    else:  
        return -1


def interpolate_and_save_poses(gt_pose_file, src_img_folder, output_path):

    gt_pose = read_csv(gt_pose_file)
    gt_pose = gt_pose.astype(float)
    img_list = sorted(os.listdir(src_img_folder), reverse=False)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        count = 1 

        for img in img_list:
            img_time = np.float64(img[:10] + '.' + img[10:-4])
            # print(img_time)
            idx = find_nearest_leftmost(np.array(gt_pose[:,0], dtype=float), img_time)

            if idx == -1:
                new_pose = gt_pose[idx+1, 1:]
                writer.writerow( np.concatenate( ([count], [img], new_pose), axis=0))


            elif idx == gt_pose.shape[0]:
                new_pose = gt_pose[idx-1, 1:]
                writer.writerow( np.concatenate( ([count], [img], new_pose), axis=0))
            
            else:
                d1 = img_time - gt_pose[idx,0]
                d2 = gt_pose[idx+1, 0] - img_time
                assert (d1 >= 0 and d2 >= 0)
                
                # position handling
                position = gt_pose[idx, 1:-3] * d2 / (d1+d2) + gt_pose[idx+1, 1:-3] * d1 / (d1+d2)

                # roll, pitch, yaw angle handling
                heading = []
                for i in range(-3, 0):
                    angle_left, angle_right = gt_pose[idx, i], gt_pose[idx+1, i]

                    # For cases where angle changes -180 -> 180 OR 180 -> -180
                    if angle_left * angle_right < 0 and abs(angle_left - angle_right) > np.float64(300) : 
                        if angle_left < 0:
                            angle_left = angle_left + np.float64(360.0)
                        elif angle_right < 0:
                            angle_right = angle_right + np.float64(360.0)

                    new_angle = angle_left  * d2 / (d1+d2) + angle_right * d1 / (d1+d2)
                    if new_angle > np.float64(180.0):
                        new_angle -= np.float64(360.0)
                    
                    heading.append(new_angle)

                writer.writerow( np.concatenate( ([count], [img], position, heading), axis=0))

            count += 1
            # if count > 300:
            #     break
    print(count, "images were processed.")


if __name__ == '__main__':
    
    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    src_root_path = config["src_root_path"]    
    dst_root_path = config["dst_root_path"]
    seq = config["seq"]
    seqs = config["all_seqs"]

    gt_filename = config["src_global_gt_filename"]
    image_folder = config["src_image_folder"]
    output_global_gt_path = config["output_global_gt_path"]

    ######### pose interpolation #########
    for seq in seqs:
        print(seq, "processing ...")
        interpolate_and_save_poses(join(src_root_path, seq, gt_filename), 
                                join(src_root_path, seq, image_folder),
                                join(dst_root_path, seq, output_global_gt_path))

    ######### read_image test #########
    # img = read_image('/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/01/thermal_left/000149.png')
    # print(img, img.shape, img.dtype, np.amax(img), np.amin(img))
    # cv2.imshow('test', img.astype('uint8'))
    # cv2.waitKey(0)

