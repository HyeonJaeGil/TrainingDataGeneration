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


def check_hard_negative(np_1, np_2, min_interval=300, min_dist=50):

        distance = ((np_1[1:4] - np_2[1:4]) ** 2).sum(0)
        if distance > min_dist:
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


def positive_sample_search_with_valid_ids(np_query, np_db, valid_ids, 
                min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    # for np_pose in np_db:
    for valid_id in valid_ids:
        is_loop = check_loop(np_query, np_db[valid_id, :])
        if is_loop:
            ids.append(int(valid_id))
    return np.array(ids)


def soft_negative_sample_search_with_valid_ids(np_query, np_db, valid_ids, 
                min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    # for np_pose in np_db:
    for valid_id in valid_ids:
        is_loop = check_soft_negative(np_query, np_db[valid_id, :])
        if is_loop:
            ids.append(int(valid_id))
    return np.array(ids)


def soft_negative_sample_search(np_query, np_db, min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    for np_pose in np_db:
        is_loop = check_soft_negative(np_query, np_pose)
        if is_loop:
            ids.append(int(np_pose[0]))
    return np.array(ids)


def generate_valid_poses(gt_pose_path, min_interval=0.1):
    
    with open(gt_pose_path, 'r') as file:
        csvreader = csv.reader(file)
        valid_ids = []
        count = 1
        prev_pose = np.zeros(3)
        
        for row in csvreader:
            if count == 1:
                prev_pose = row[2:5]
                valid_ids.append(row[0])
                count += 1    
            else:
                interval = ((np.array(prev_pose, dtype="float64") - np.array(row[2:5], dtype="float64"))**2).sum(0)
                if interval > min_interval ** 2:
                    valid_ids.append(row[0])
                    prev_pose = row[2:5]
                count += 1

        return np.array(valid_ids, dtype='int')


def generate_intra_loop_pairs(gt_pose_path, output_path, seq, max_dist=3):

    pose_array = read_interpolated_pose(gt_pose_path)
    valid_poses_ids = generate_valid_poses(gt_pose_path)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        
        count = 0
        # for pose in pose_array:
        for pose_id in valid_poses_ids:
            count += 1
            query = pose_array[pose_id-1, :]
            # candidates = positive_sample_search(query, pose_array)
            candidates = positive_sample_search_with_valid_ids(query, pose_array, valid_poses_ids)
            for candidate in candidates:
                writer.writerow([query[0], candidate, seq, seq, 1])
            print(count, candidates.shape)
        
        print(candidates, candidates.shape)


def generate_inter_loop_pairs(query_gt_path, search_gt_paths, seq1, seqs, output_path, max_dist=3):
    
    pose_array1 = read_interpolated_pose(query_gt_path)
    valid_poses_ids1 = generate_valid_poses(query_gt_path)

    # pose_array2 = read_interpolated_pose(gt_pose_path2)
    # valid_poses_ids2 = generate_valid_poses(gt_pose_path2)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(search_gt_paths)):
            pose_array2 = read_interpolated_pose(search_gt_paths[i])
            valid_poses_ids2 = generate_valid_poses(search_gt_paths[i])
            search_seq = seqs[i]

            count = 0
            for pose_id in valid_poses_ids1:
                query = pose_array1[pose_id-1, :]
                # candidates = positive_sample_search(query, pose_array2)
                candidates = positive_sample_search_with_valid_ids(query, pose_array2, valid_poses_ids2)
                for candidate in candidates:
                    writer.writerow([query[0], candidate, seq1, search_seq, 1])
                    count += 1
            print('finished seq number: ', i)
            print('total loop count: ', count)


if __name__ == '__main__':

    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    dst_root_path = config["dst_root_path"]
    seq = config["seq"]
    # dst_root_path = os.path.join(dst_root_path, seq)
    search_seqs = ["07", "08", "09"]

    interpolated_gt_path = config["interpolated_gt_path"]
    intraloop_save_path = config["intraloop_save_path"]
    interloop_save_path = config["interloop_save_path"] 
    abs_interpolated_gt_path = os.path.join(dst_root_path, seq, interpolated_gt_path)
    abs_intraloop_save_path = os.path.join(dst_root_path, seq, intraloop_save_path)
    abs_interloop_save_path = os.path.join(dst_root_path, seq, interloop_save_path)

    valid_pose = generate_valid_poses(abs_interpolated_gt_path)
    print(valid_pose.shape)
    # generate_intra_loop_pairs(abs_interpolated_gt_path, abs_intraloop_save_path, seq=seq)

    abs_interloop_search_paths = []
    for search_seq in search_seqs:
        abs_interloop_search_path = os.path.join(dst_root_path, search_seq, interpolated_gt_path)
        abs_interloop_search_paths.append(abs_interloop_search_path)
    
    generate_inter_loop_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
                                seq, search_seqs, abs_interloop_save_path)

    # pose_array = read_interpolated_pose(interpolated_gt_path)
    # for i in range(9580, 9600):
        # print(check_loop(pose_array[5300,:], pose_array[i,:]))
    
