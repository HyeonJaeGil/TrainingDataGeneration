from itertools import count
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
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import csv
import yaml
import math
from utils import read_csv, read_interpolated_pose
np.set_printoptions(precision=9)


def check_true(p=0.5):
    if np.random.rand() <= p:
        return True
    else: 
        return False


def check_loop(np_1, np_2, min_interval=300, max_dist=5, heading_tolerance = 30):

        distance = ((np_1[1:4] - np_2[1:4]) ** 2).sum(0)
        id_interval = abs(np_1[0] - np_2[0])
        heading_differnce = ((np_1[4:] - np_2[4:]) ** 2).sum(0)
        if distance < max_dist**2 and id_interval > min_interval and heading_differnce < heading_tolerance**2:
            return True
        else:
            return False


def check_soft_negative(np_1, np_2, min_dist=6, max_dist=20.):

        distance = ((np_1 - np_2) ** 2).sum(0)
        if distance < max_dist**2 and distance > min_dist**2:
            return True
        else:
            return False


def check_hard_negative(np_1, np_2, min_dist=50.):

        distance = ((np_1 - np_2) ** 2).sum(0)
        if distance > (min_dist**2):
            return True
        else:
            return False


def get_valid_ids(gt_pose_path, min_interval=1):
    
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


def get_valid_image_dict(gt_pose_path, min_interval=0.1):
    
    with open(gt_pose_path, 'r') as file:
        csvreader = csv.reader(file)
        valid_ids = []
        valid_imgs = []
        count = 1
        prev_pose = np.zeros(3)
        
        for row in csvreader:
            if count == 1:
                prev_pose = row[2:5]
                valid_ids.append(row[0])
                valid_imgs.append(row[1])
                count += 1    
            else:
                interval = ((np.array(prev_pose, dtype="float64") - np.array(row[2:5], dtype="float64"))**2).sum(0)
                if interval > min_interval ** 2:
                    valid_ids.append(row[0])
                    valid_imgs.append(row[1])
                    prev_pose = row[2:5]
                count += 1

        np_ids = np.reshape(np.array(valid_ids, dtype=str), (-1, 1))
        np_imgs = np.reshape(np.array(valid_imgs, dtype=str), (-1, 1))

        return np.concatenate((np_ids, np_imgs), axis=1)


def get_valid_poses(query_gt_path, min_interval=1):

    with open(query_gt_path, 'r') as file:
        csvreader = csv.reader(file)
        valid_ids = []
        valid_poses = []
        count = 1
        prev_pose = np.zeros(3)
        
        for row in csvreader:
            if count == 1:
                prev_pose = row[2:5]
                valid_ids.append(row[0])
                valid_poses.append(row[-6:])
                count += 1    
            else:
                interval = ((np.array(prev_pose, dtype="float64")
                            - np.array(row[2:5], dtype="float64"))**2).sum(0)
                if interval > min_interval ** 2:
                    valid_ids.append(row[0])
                    valid_poses.append(row[-6:])
                    prev_pose = row[2:5]
                count += 1
        
        np_ids = np.reshape(np.array(valid_ids, dtype=int), (-1, 1))
        np_imgs = np.reshape(np.array(valid_poses, dtype=float), (-1, 6))
        return np.concatenate((np_ids, np_imgs), axis=1)


def save_valid_poses(gt_pose_path, save_path, min_interval=1):
    
    with open(gt_pose_path, 'r') as input_f, open(save_path, 'w') as output_f:
        csvreader = csv.reader(input_f)
        writer = csv.writer(output_f)

        valid_ids = []
        valid_poses = []
        count = 1
        prev_pose = np.zeros(3)
        
        for row in csvreader:
            if count == 1:
                writer.writerow([row[0], row[2], row[3], row[4]])
                prev_pose = row[2:5]

                count += 1    
            else:
                interval = ((np.array(prev_pose, dtype="float64")
                            - np.array(row[2:5], dtype="float64"))**2).sum(0)
                if interval > min_interval ** 2:
                    writer.writerow([row[0], row[2], row[3], row[4]])
                    prev_pose = row[2:5]
                    count += 1
        
    print('valid poses number: ', count)


def count_samples(query_gt_path, search_gt_path, query_seq, search_seq, 
                min_interval=1, min_id_diff=300, pos_max_dist=5, max_heading_diff=30, 
                softneg_min_dist=6, softneg_max_dist=20, hardneg_min_dist=50,  
                p_loop=1.0, p_softneg=1.0, p_hardneg=0.0, p_dropout=1.0):

    q_poses = get_valid_poses(query_gt_path, min_interval = min_interval)
    q_indexes = q_poses[:, 0]
    q_positions = q_poses[:, 1:4]
    q_headings = q_poses[:, 4:]

    count = 0
    total_pos_samples, total_softneg_samples, total_hardneg_samples = 0, 0, 0
    s_poses = get_valid_poses(search_gt_path, min_interval = min_interval)
    s_indexes = s_poses[:, 0]
    s_positions = s_poses[:, 1:4]
    s_headings = s_poses[:, 4:]
    
    pose_tree = KDTree(s_positions)

    for q_idx in range(q_positions.shape[0]):
        pos_samples, softneg_samples, hardneg_samples = 0, 0, 0
        
        # loop search
        if check_true(p_loop):
            for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=pos_max_dist, p=2):
                heading_diff = ((q_headings[q_idx,:]-s_headings[idx,:])**2).sum()
                id_diff = abs(q_indexes[q_idx]-s_indexes[idx])

                # intra loop search
                if seq1 == seq2 and check_true(p_dropout) and \
                heading_diff < max_heading_diff**2 and id_diff > min_id_diff:
                    pos_samples += 1
                    count += 1
                    
                # inter loop search    
                if seq1 != seq2 and check_true(p_dropout) and \
                heading_diff < max_heading_diff**2:
                    pos_samples += 1
                    count += 1
        
        # soft-negative search
        if check_true(p_softneg) and pos_samples > 0:
            for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=softneg_max_dist, p=2):
                if check_true(p_dropout) and check_soft_negative \
                (q_positions[q_idx,:], s_positions[idx,:], softneg_min_dist, softneg_max_dist):
                    softneg_samples += 1
                    count += 1
        
        # hard-negative search
        if check_true(p_hardneg) and pos_samples > 0:
            for idx in np.random.permutation(s_positions.shape[0]):
                if check_true(p_dropout) and check_hard_negative \
                (q_positions[q_idx,:], s_positions[idx,:], hardneg_min_dist):
                    hardneg_samples += 1
                    count += 1
        
        total_pos_samples += pos_samples
        total_softneg_samples += softneg_samples
        total_hardneg_samples += hardneg_samples
    
    print("query seq:", query_seq, "search seq:", search_seq)
    print("Total positive sample: ", total_pos_samples, 
        "total softneg samples: ", total_softneg_samples, 
        "total hardneg samples: ", total_hardneg_samples)
    print("Total sample counts: ", count)



def generate_inter_loop_pairs_kdtree(query_gt_path, search_gt_paths, seq1, seqs, output_path, 
                        pos_max_dist=5, max_heading_diff=30, softneg_min_dist=6, softneg_max_dist=20, min_id_diff=300):

    query_pose_array = get_valid_poses(query_gt_path)
    query_index_array = query_pose_array[:, 0]
    query_position_array = query_pose_array[:, 1:4]
    query_heading_array = query_pose_array[:, 4:]

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(search_gt_paths)):
            count = 0
            search_seq = seqs[i]
            search_pose_array = get_valid_poses(search_gt_paths[i])
            search_index_array = search_pose_array[:, 0]
            search_position_array = search_pose_array[:, 1:4]
            search_heading_array = search_pose_array[:, 4:]
            
            pose_tree = KDTree(search_position_array)
            heading_tree = KDTree(search_heading_array)

            for query_idx in range(query_position_array.shape[0]):
                for idx in pose_tree.query_ball_point(query_position_array[query_idx, :], r=pos_max_dist, p=2):
                # for idx in idxs:
                    if ((query_heading_array[query_idx,:]-search_heading_array[idx,:])**2).sum() < max_heading_diff**2 \
                    and abs(query_index_array[query_idx]-search_index_array[idx]) > min_id_diff:
                        writer.writerow([query_index_array[query_idx],search_index_array[idx], seq1, search_seq, 1])
                        count += 1
            
            print("query seq:", seq1, "search seq:", search_seq)
            print("loop count: ", count)



'''
kaist: 0.5, 0.4, 0.2, 0.2, 50, 20, 10 
snu: 0.7, 0.2, 0.1
valley: 0.5, 0.3, 0.1
'''

def generate_total_pairs_kdtree(query_gt_path, search_gt_path, query_seq, search_seq, output_path, 
                min_interval=1, min_id_diff=300, pos_max_dist=5, max_heading_diff=30, 
                softneg_min_dist=6, softneg_max_dist=20, hardneg_min_dist=50,                      
                p_loop=1.0, p_softneg=1.0, p_hardneg=0.0, p_dropout=1.0,
                max_pos_samples=50, max_softneg_samples=20, max_hardneg_samples=10):

    with open(output_path, 'a') as f:
        writer = csv.writer(f)

        q_poses = get_valid_poses(query_gt_path, min_interval = min_interval)
        q_indexes = q_poses[:, 0]
        q_positions = q_poses[:, 1:4]
        q_headings = q_poses[:, 4:]

        count = 0
        total_pos_samples, total_softneg_samples, total_hardneg_samples = 0, 0, 0
        s_poses = get_valid_poses(search_gt_path, min_interval = min_interval)
        s_indexes = s_poses[:, 0]
        s_positions = s_poses[:, 1:4]
        s_headings = s_poses[:, 4:]
        
        pose_tree = KDTree(s_positions)

        for q_idx in range(q_positions.shape[0]):
            pos_samples, softneg_samples, hardneg_samples = 0, 0, 0
            
            # loop search
            if check_true(p_loop):
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=pos_max_dist, p=2):
                    heading_diff = ((q_headings[q_idx,:]-s_headings[idx,:])**2).sum()
                    id_diff = abs(q_indexes[q_idx]-s_indexes[idx])

                    # intra loop search
                    if query_seq == search_seq and check_true(p_dropout) and \
                    heading_diff < max_heading_diff**2 and id_diff > min_id_diff:
                        writer.writerow([q_indexes[q_idx], s_indexes[idx], query_seq, search_seq, 1])
                        pos_samples += 1
                        count += 1
                        
                    # inter loop search    
                    if query_seq != search_seq and check_true(p_dropout) and \
                    heading_diff < max_heading_diff**2:
                        writer.writerow([q_indexes[q_idx], s_indexes[idx], query_seq, search_seq, 1])
                        pos_samples += 1
                        count += 1
            
            # soft-negative search
            if check_true(p_softneg):
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=softneg_max_dist, p=2):
                    if check_true(p_dropout) and check_soft_negative \
                    (q_positions[q_idx,:], s_positions[idx,:], softneg_min_dist, softneg_max_dist):
                        writer.writerow([q_indexes[q_idx], s_indexes[idx], query_seq, search_seq, 0.5])
                        softneg_samples += 1
                        count += 1
            
            # hard-negative search
            if check_true(p_hardneg):
                for idx in np.random.permutation(s_positions.shape[0]):
                    if check_true(p_dropout) and check_hard_negative \
                    (q_positions[q_idx,:], s_positions[idx,:], hardneg_min_dist):
                        writer.writerow([q_indexes[q_idx], s_indexes[idx], query_seq, search_seq, 0])
                        hardneg_samples += 1
                        count += 1
            
            total_pos_samples += pos_samples
            total_softneg_samples += softneg_samples
            total_hardneg_samples += hardneg_samples
        
        print("query seq:", query_seq, "search seq:", search_seq)
        print("Total positive sample: ", total_pos_samples, 
            "total softneg samples: ", total_softneg_samples, 
            "total hardneg samples: ", total_hardneg_samples)
        print("Total sample counts: ", count)


def remove_prev_samples(root_path, seqs, total_save_path):
    for seq in seqs:
        train_pairs_filepath = join(root_path, seq, total_save_path)
        if os.path.exists(train_pairs_filepath):
            os.remove(train_pairs_filepath)
            print("removed previous", train_pairs_filepath)


if __name__ == '__main__':

    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    
    all_seqs = config["all_seqs"] 
    seq = config["seq"]
    
    dst_root_path = config["dst_root_path"]
    interpolated_gt_path = config["interpolated_gt_path"]
    total_save_path = config["total_save_path"] 
    valid_pose_save_path = config["valid_pose_save_path"] 

# ["01", "02", "03"]
# ["04", "05", "06"]
# ["07", "08", "09"]

    pairs = [
        ["01", "01"], ["01", "02"], ["01", "03"],
        ["02", "01"], ["02", "02"], ["02", "03"],
        ["03", "01"], ["03", "02"], ["03", "03"], 
        ["04", "04"], ["04", "05"], ["04", "06"],
        ["05", "04"], ["05", "05"], ["05", "06"],
        ["06", "04"], ["06", "05"], ["06", "06"], 
        ["07", "07"], ["07", "08"], ["07", "09"],
        ["08", "07"], ["08", "08"], ["08", "09"],
        ["09", "07"], ["09", "08"], ["09", "09"], 
        ["10", "10"], ["11", "11"]
    ]    


    # # pose_array = read_interpolated_pose(interpolated_gt_path)
    # # for i in range(9580, 9600):
    #     # print(check_loop(pose_array[5300,:], pose_array[i,:]))
    
    min_interval = 1
    print("min interval: ", min_interval)

    remove_prev_samples(dst_root_path, all_seqs, total_save_path)

    for seq1, seq2 in pairs:
        abs_query_gt_path = join(dst_root_path, seq1, interpolated_gt_path)
        abs_total_save_path = join(dst_root_path, seq1, total_save_path)
        abs_valid_pose_save_path = join(dst_root_path, seq1, 'train_sets/valid_poses.csv')
        abs_search_gt_path = join(dst_root_path, seq2, interpolated_gt_path)
        
        # save_valid_poses(abs_query_gt_path, abs_valid_pose_save_path, min_interval=min_interval)
        
        # count_samples(abs_query_gt_path, abs_search_gt_path, 
        #                             seq1, seq2, min_interval=min_interval)
        
        generate_total_pairs_kdtree(join(dst_root_path, seq1, interpolated_gt_path), 
                                    join(dst_root_path, seq2, interpolated_gt_path), 
                                    seq1, seq2, 
                                    join(dst_root_path, seq1, total_save_path),
                                    min_interval=min_interval)
        
        ###### save only loop #####
        # generate_total_pairs_kdtree(join(dst_root_path, seq1, interpolated_gt_path), 
        #                             join(dst_root_path, seq2, interpolated_gt_path), 
        #                             seq1, seq2, 
        #                             join(dst_root_path, seq1, "train_sets/pos_pairs.csv"),
        #                             min_interval=min_interval,
        #                             p_softneg=0.0, p_hardneg=0.0)        
        
        ###### save only negative pair ######
        # generate_total_pairs_kdtree(join(dst_root_path, seq1, interpolated_gt_path), 
        #                             join(dst_root_path, seq2, interpolated_gt_path), 
        #                             seq1, seq2, 
        #                             join(dst_root_path, seq1, "train_sets/neg_pairs.csv"),
        #                             min_interval=min_interval,
        #                             p_loop = 0.0)


        print("")