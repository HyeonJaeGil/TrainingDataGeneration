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
np.set_printoptions(precision=9)

# read csv file that contains local pose
def read_pose(file_path): 

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
    
    return np.asarray(rows, dtype='float64')


# read csv file that contains local (interpolated) pose
def read_interpolated_pose(file_path): 

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append([row[0]] + row[2:])
    
    return np.asarray(rows, dtype='float64')


# read csv file that contains local pose
def read_pairs(file_path): 

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
    
    return np.asarray(rows, dtype='float64')


def read_files_from_path(file_path):

    for file in os.listdir(file_path):
        data = np.array(cv2.imread(os.path.join(file_path,file)))
        data = os.path.join(file_path,file)
        print(data)


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
    # print(idx)

    if idx > 0 and value - float(array[idx]) <= 0:
        return idx-1
    else: 
        return -1


def construct_gt_pose_array(image_folder, pose_table, output_path):

    img_list = sorted(os.listdir(image_folder), reverse=False)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        count = 1 

        for img in img_list:
            img_time = np.float64(img[:10] + '.' + img[10:-4])
            print(img_time)
            idx = find_nearest_leftmost(pose_table[:,0], img_time)

            if idx == -1:
                new_pose = pose_table[idx+1, 1:]
                writer.writerow( np.concatenate( ([count], [img], new_pose), axis=0))


            elif idx == pose_table.shape[0]:
                new_pose = pose_table[idx-1, 1:]
                writer.writerow( np.concatenate( ([count], [img], new_pose), axis=0))
            
            else:
                d1 = img_time - pose_table[idx,0]
                d2 = pose_table[idx+1, 0] - img_time
                assert (d1 >= 0 and d2 >= 0)
                
                # position handling
                new_pose = pose_table[idx, 1:-3] * d2 / (d1+d2) + pose_table[idx+1, 1:-3] * d1 / (d1+d2)

                # yaw angle handling
                if pose_table[idx, -1] * pose_table[idx+1, -1] < 0 and abs(pose_table[idx, -1] - pose_table[idx+1, -1]) > np.float64(300) : 
                    new_yaw_1 = pose_table[idx, -1] + np.float64(360.0)
                else: 
                    new_yaw_1 = pose_table[idx, -1]
                if pose_table[idx+1, -1] * pose_table[idx+1, -1] < 0 and abs(pose_table[idx, -1] - pose_table[idx+1, -1]) > np.float64(300) : 
                    new_yaw_2 = pose_table[idx+1, -1] + np.float64(360.0)
                else: 
                    new_yaw_2 = pose_table[idx+1, -1]
                interpolated_yaw = new_yaw_1  * d2 / (d1+d2) + new_yaw_2 * d1 / (d1+d2)
                
                if interpolated_yaw > np.float64(180.0):
                    interpolated_yaw -= np.float64(360.0)

                # pitch angle handling
                if pose_table[idx, -2] * pose_table[idx+1, -2] < 0 and abs(pose_table[idx, -2] - pose_table[idx+1, -2]) > np.float64(300) : 
                    new_pitch_1 = pose_table[idx, -2] + np.float64(360.0)
                else: 
                    new_pitch_1 = pose_table[idx, -2]
                if pose_table[idx+1, -2] * pose_table[idx+1, -2] < 0 and abs(pose_table[idx, -2] - pose_table[idx+1, -2]) > np.float64(300) : 
                    new_pitch_2 = pose_table[idx+1, -2] + np.float64(360.0)
                else: 
                    new_pitch_2 = pose_table[idx+1, -2]
                interpolated_pitch = new_pitch_1  * d2 / (d1+d2) + new_pitch_2 * d1 / (d1+d2)
                
                if interpolated_pitch > np.float64(180.0):
                    interpolated_pitch -= np.float64(360.0)

                # roll angle handling
                if pose_table[idx, -3] * pose_table[idx+1, -3] < 0 and abs(pose_table[idx, -3] - pose_table[idx+1, -3]) > np.float64(300) : 
                    new_roll_1 = pose_table[idx, -3] + np.float64(360.0)
                else: 
                    new_roll_1 = pose_table[idx, -3]
                if pose_table[idx+1, -3] * pose_table[idx+1, -3] < 0 and abs(pose_table[idx, -3] - pose_table[idx+1, -3]) > np.float64(300) : 
                    new_roll_2 = pose_table[idx+1, -3] + np.float64(360.0)
                else: 
                    new_roll_2 = pose_table[idx+1, -3]
                interpolated_roll = new_roll_1  * d2 / (d1+d2) + new_roll_2 * d1 / (d1+d2)
                
                if interpolated_roll > np.float64(180.0):
                    interpolated_roll -= np.float64(360.0)

                writer.writerow( np.concatenate( ([count], [img], new_pose, 
                            [interpolated_roll, interpolated_pitch, interpolated_yaw]), axis=0))

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
    src_root_path = os.path.join(src_root_path, seq)
    dst_root_path = os.path.join(dst_root_path, seq)
    image_folder = config["image_folder"]
    pose_folder = config["pose_folder"]
    output_local_gt_path = config["output_local_gt_path"]
    output_global_gt_path = config["output_global_gt_path"]
    
    abs_image_folder = os.path.join(src_root_path, image_folder)
    abs_pose_folder = os.path.join(src_root_path, pose_folder)    
    abs_output_local_gt_path = os.path.join(dst_root_path, output_local_gt_path)
    abs_output_global_gt_path = os.path.join(dst_root_path, output_global_gt_path)

    # local_pose_table = read_pose(os.path.join(abs_pose_folder, 'local_pose.csv'))
    global_pose_table = read_pose(os.path.join(abs_pose_folder, 'global_pose.csv'))
    # construct_gt_pose_array(abs_image_folder, local_pose_table, abs_output_local_gt_path)
    # construct_gt_pose_array(abs_image_folder, global_pose_table, abs_output_global_gt_path)

    # query = np.float64([-1.4168655872345, -0.863484025001526, 0.398936808109283])
    # print(query)
    # print(np_pose_table.shape[0])
    # for pose in np_pose_table:
    #     xyz = pose[1:4]
    #     print(((xyz-query)**2).sum(0))
    #     break

    # pose_table = read_pose(output_gt_path)
    # show_loop_images(pose_table, image_folder, 5317, 9602)
    # with open(output_gt_path, 'r') as file:
    #     csvreader = csv.reader(file)
    #     show_loop_images(f, )

    img = read_image('/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/01/thermal_left/000149.png')
    print(img, img.shape, img.dtype, np.amax(img), np.amin(img))
    cv2.imshow('test', img.astype('uint8'))
    cv2.waitKey(0)

    # img = read_image('/media/hj/seagate/datasets/STHEREO/01/image/stereo_thermal_14_left/1630106835318041837.png')
    # print(img, img.shape, img.dtype, np.amax(img), np.amin(img))
    # cv2.imshow('test', img.astype('uint8'))
    # cv2.waitKey(0)
