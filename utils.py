from operator import gt
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import numpy as np
import cv2
import matplotlib as plt
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



def read_files_from_path(file_path):

    for file in os.listdir(file_path):
        data = np.array(cv2.imread(os.path.join(file_path,file)))
        data = os.path.join(file_path,file)
        print(data)


def find_nearest_leftmost(array,value):
    
    if np.min(array) > value:
        return -1
    elif np.max(array) < value:
        return len(array)

    idx = np.searchsorted(array, value, side="left")
    print(idx)

    if idx > 0 and value - float(array[idx]) <= 0:
        return idx-1
    else: 
        return -1


def construct_gt_pose_array(img_list, pose_table, output_path):

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        count = 0 

        for img in img_list:
            img_time = np.float64(img[:10] + '.' + img[10:-4])
            print(img_time)
            idx = find_nearest_leftmost(pose_table[:,0], img_time)

            if idx == -1:
                new_pose = pose_table[idx+1, 1:]

            elif idx == pose_table.shape[0]:
                new_pose = pose_table[idx-1, 1:]
            
            else:
                d1 = img_time - pose_table[idx,0]
                d2 = pose_table[idx+1, 0] - img_time
                assert (d1 >= 0 and d2 >= 0)
                new_pose = pose_table[idx, 1:] * d2 / (d1+d2) + pose_table[idx+1, 1:] * d1 / (d1+d2)

            # writer.writerow( np.concatenate( ([img[:-4]], new_pose), axis=0))
            writer.writerow( np.concatenate( ([img_time], new_pose), axis=0))

            # count += 1
            # if count > 400:
            #     break


if __name__ == '__main__':
    
    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    image_folder = config["image_folder"]
    pose_folder = config["pose_folder"]
    output_gt_path = config["output_gt_path"]
    
    np_pose_table = read_pose(os.path.join(pose_folder, 'local_pose.csv'))
    # print(np_pose_table, np_pose_table.shape)
    # print(np_pose_table[:,0])
    # result = find_nearest_leftmost(np_pose_table[:,0], np.float64(1630132166.0))
    # print(result)
    '''
    construct array [img_filename, x, y, z, r, p, y]
    '''
    
    # print(np_pose_table[0,:], np_pose_table[:,1].shape)
    img_list = sorted(os.listdir(image_folder), reverse=False)
    # print(img_list)
    construct_gt_pose_array(img_list, np_pose_table, output_gt_path)
