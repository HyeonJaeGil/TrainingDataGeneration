from operator import gt
import os
import sys
from tabnanny import check
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
from utils import read_pose, read_interpolated_pose
np.set_printoptions(precision=9)


def check_loop(np_1, np_2, min_interval=300, max_dist=5, heading_tolerance = 900):

        distance = ((np_1[1:4] - np_2[1:4]) ** 2).sum(0)
        id_interval = abs(np_1[0] - np_2[0])
        heading_differnce = ((np_1[4:] - np_2[4:]) ** 2).sum(0)
        if distance < max_dist and id_interval > min_interval and heading_differnce < heading_tolerance:
            return True
        else:
            return False


def check_soft_negative(np_1, np_2, min_interval=300, min_dist=6, max_dist=20):

        distance = ((np_1[1:4] - np_2[1:4]) ** 2).sum(0)
        if distance < max_dist and distance > min_dist:
            return True
        else:
            return False


def positive_sample_search(np_query, np_db, min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    for np_pose in np_db:
        is_loop = check_loop(np_query, np_pose)
        if is_loop:
            ids.append(int(np_pose[0]))
    return np.array(ids)


def generate_intra_loop_pairs(gt_pose_path, output_path, seq=2, max_dist=3):

    pose_array = read_interpolated_pose(gt_pose_path)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
    
        count = 0
        for pose in pose_array:
            count += 1
            # if count % 100 == 0:
            if True:
                query = pose
                candidates = positive_sample_search(query, pose_array)
                print(candidates.shape)
                for candidate in candidates:
                    writer.writerow([query[0], candidate, seq, seq, 1])
            
                print(count)
            # break
        print(candidates, candidates.shape)
        

def generate_inter_loop_pairs(gt_pose_path1, gt_pose_path2, output_path, max_dist=3):
    pass


if __name__ == '__main__':

    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    interpolated_gt_path = config["interpolated_gt_path"]
    intraloop_save_path = config["intraloop_save_path"]
    generate_intra_loop_pairs(interpolated_gt_path, intraloop_save_path, seq=1)
    
    # pose_array = read_interpolated_pose(interpolated_gt_path)
    # for i in range(9580, 9600):
        # print(check_loop(pose_array[5300,:], pose_array[i,:]))
    
