from itertools import count
from operator import gt
import os
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
from utils import read_pose, read_interpolated_pose
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


def generate_valid_poses(gt_pose_path, min_interval=1):
    
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


def generate_valid_image_dictionary(gt_pose_path, min_interval=0.1):
    
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
                interval = ((np.array(prev_pose, dtype="float64") - np.array(row[2:5], dtype="float64"))**2).sum(0)
                if interval > min_interval ** 2:
                    writer.writerow([row[0], row[2], row[3], row[4]])
                    prev_pose = row[2:5]
                    count += 1
        
    
    print('valid poses number: ', count)



def generate_filtered_pose_array(query_gt_path, min_interval = 1):
    
    pose_array1 = read_interpolated_pose(query_gt_path)
    valid_poses_ids1 = generate_valid_poses(query_gt_path, min_interval= min_interval)

    filtered_array = np.array([], dtype="float").reshape(0,pose_array1.shape[-1])
    for id in valid_poses_ids1:
        filtered_array = np.vstack([filtered_array, pose_array1[id-1, :]])

    # print(filtered_array, filtered_array.shape)
    return filtered_array



def count_samples(query_gt_path, search_gt_path, query_seq, search_seq, 
                min_interval=1, min_id_diff=300, pos_max_dist=5, max_heading_diff=30, 
                softneg_min_dist=6, softneg_max_dist=20, hardneg_min_dist=50,  
                p_loop=1.0, p_softneg=1.0, p_hardneg=0.0, p_dropout=1.0):

    q_poses = generate_filtered_pose_array(query_gt_path, min_interval = min_interval)
    q_indexes = q_poses[:, 0]
    q_positions = q_poses[:, 1:4]
    q_headings = q_poses[:, 4:]

    count = 0
    total_pos_samples, total_softneg_samples, total_hardneg_samples = 0, 0, 0
    s_poses = generate_filtered_pose_array(search_gt_path, min_interval = min_interval)
    s_indexes = s_poses[:, 0]
    s_positions = s_poses[:, 1:4]
    s_headings = s_poses[:, 4:]
    
    pose_tree = KDTree(s_positions)

    for q_idx in range(q_positions.shape[0]):
        pos_samples, softneg_samples, hardneg_samples = 0, 0, 0
        
        if seq1 == seq2: # intra loop search
            if check_true(p_loop):
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=pos_max_dist, p=2):
                    heading_diff = ((q_headings[q_idx,:]-s_headings[idx,:])**2).sum()
                    id_diff = abs(q_indexes[q_idx]-s_indexes[idx])
                    if check_true(p_dropout) and heading_diff < max_heading_diff**2 and id_diff > min_id_diff:
                        pos_samples += 1
                        count += 1
            if check_true(p_softneg) and pos_samples > 0:
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=softneg_max_dist, p=2):
                    if check_true(p_dropout) and check_soft_negative \
                    (q_positions[q_idx,:], s_positions[idx,:], softneg_min_dist, softneg_max_dist):
                        softneg_samples += 1
                        count += 1
            
            if check_true(p_hardneg) and pos_samples > 0:
                for idx in np.random.permutation(s_positions.shape[0]):
                    if check_true(p_dropout) and check_hard_negative(q_positions[q_idx,:], s_positions[idx,:], hardneg_min_dist):
                        hardneg_samples += 1
                        count += 1
        
        elif seq1 != seq2: # inter loop search    
            if check_true(p_loop):
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=pos_max_dist, p=2):
                    heading_diff = ((q_headings[q_idx,:]-s_headings[idx,:])**2).sum()
                    if check_true(p_dropout) and heading_diff < max_heading_diff**2:
                        pos_samples += 1
                        count += 1
            if check_true(p_softneg) and pos_samples > 0:
                for idx in pose_tree.query_ball_point(q_positions[q_idx, :], r=softneg_max_dist, p=2):
                    if check_true(p_dropout) and check_soft_negative \
                    (q_positions[q_idx,:], s_positions[idx,:], softneg_min_dist, softneg_max_dist):
                        softneg_samples += 1
                        count += 1
            
            if check_true(p_hardneg) and pos_samples > 0:
                for idx in np.random.permutation(s_positions.shape[0]):
                    if check_true(p_dropout) and check_hard_negative(q_positions[q_idx,:], s_positions[idx,:], hardneg_min_dist):
                        hardneg_samples += 1
                        count += 1

        total_pos_samples += pos_samples
        total_softneg_samples += softneg_samples
        total_hardneg_samples += hardneg_samples
    
    print("query seq:", query_seq, "search seq:", search_seq)
    # print("positive sample: ", pos_samples, "softneg samples: ", softneg_samples, "hardneg_samples: ", hardneg_samples)
    print("Total positive sample: ", total_pos_samples, "total softneg samples: ", total_softneg_samples, "total hardneg samples: ", total_hardneg_samples)
    print("Total sample counts: ", count)





def generate_inter_loop_pairs_kdtree(query_gt_path, search_gt_paths, seq1, seqs, output_path, 
                        pos_max_dist=5, max_heading_diff=30, softneg_min_dist=6, softneg_max_dist=20, min_id_diff=300):

    query_pose_array = generate_filtered_pose_array(query_gt_path)
    query_index_array = query_pose_array[:, 0]
    query_position_array = query_pose_array[:, 1:4]
    query_heading_array = query_pose_array[:, 4:]

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(search_gt_paths)):
            count = 0
            search_seq = seqs[i]
            search_pose_array = generate_filtered_pose_array(search_gt_paths[i])
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

def generate_total_pairs_kdtree(query_gt_path, search_gt_paths, seq1, search_seqs, output_path, hardneg_min_dist=50,
                    pos_max_dist=5, max_heading_diff=30, softneg_min_dist=6, softneg_max_dist=20, min_id_diff=300, 
                    p_loop=0.5, p_softneg=0.5, p_hardneg=0.3, p_dropout=0.4,
                    max_pos_samples=50, max_softneg_samples=20, max_hardneg_samples=10):

    q_pose_array = generate_filtered_pose_array(query_gt_path)
    q_index_array = q_pose_array[:, 0]
    q_position_array = q_pose_array[:, 1:4]
    q_heading_array = q_pose_array[:, 4:]


    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(search_gt_paths)):
            count = 0
            total_pos_samples, total_softneg_samples, total_hardneg_samples = 0, 0, 0
            search_seq = search_seqs[i]
            s_pose_array = generate_filtered_pose_array(search_gt_paths[i])
            s_index_array = s_pose_array[:, 0]
            s_position_array = s_pose_array[:, 1:4]
            s_heading_array = s_pose_array[:, 4:]
            
            pose_tree = KDTree(s_position_array)

            for q_idx in range(q_position_array.shape[0]):
                pos_samples, softneg_samples, hardneg_samples = 0, 0, 0
                
                if check_true(p_loop): # 50% probability choose
                    for idx in pose_tree.query_ball_point(q_position_array[q_idx, :], r=pos_max_dist, p=2):
                    # for idx in idxs:
                        if check_true(p_dropout) and \
                        ((q_heading_array[q_idx,:]-s_heading_array[idx,:])**2).sum() < max_heading_diff**2 and \
                         (abs(q_index_array[q_idx]-s_index_array[idx]) > min_id_diff):
                            writer.writerow([q_index_array[q_idx],s_index_array[idx], seq1, search_seq, 1])
                            pos_samples += 1
                            count += 1
                        if pos_samples >= max_pos_samples: break
                if check_true(p_softneg) and pos_samples > 0:
                # if pos_samples > 0:
                    for idx in pose_tree.query_ball_point(q_position_array[q_idx, :], r=softneg_max_dist, p=2):
                    # candidates = np.array(pose_tree.query_ball_point(q_position_array[q_idx, :], r=softneg_max_dist, p=2))
                    # print(candidates.shape)
                    # for idx in np.random.permutation(candidates.shape[0]):
                        if check_true(p_dropout) and check_soft_negative \
                        (q_position_array[q_idx,:], s_position_array[idx,:], softneg_min_dist, softneg_max_dist):
                            writer.writerow([q_index_array[q_idx],s_index_array[idx], seq1, search_seq, 0.5])
                            softneg_samples += 1
                            count += 1
                        if softneg_samples >= max_softneg_samples or pos_samples*p_softneg <= softneg_samples: break
               
                if check_true(p_hardneg) and pos_samples > 0:
                    for idx in np.random.permutation(s_position_array.shape[0]):
                        if check_true(p_dropout) and check_hard_negative(q_position_array[q_idx,:], s_position_array[idx,:], hardneg_min_dist):
                            writer.writerow([q_index_array[q_idx],s_index_array[idx], seq1, search_seq, 0.0])
                            hardneg_samples += 1
                            count += 1
                        if hardneg_samples >= max_hardneg_samples or pos_samples*p_hardneg <= hardneg_samples: break

                total_pos_samples += pos_samples
                total_softneg_samples += softneg_samples
                total_hardneg_samples += hardneg_samples
            
            print("query seq:", seq1, "search seq:", search_seq)
            # print("positive sample: ", pos_samples, "softneg samples: ", softneg_samples, "hardneg_samples: ", hardneg_samples)
            print("Total positive sample: ", total_pos_samples, "total softneg samples: ", total_softneg_samples, "total hardneg samples: ", total_hardneg_samples)
            print("Total sample counts: ", count)



if __name__ == '__main__':

    config_filename = './config/config.yaml'
    config = yaml.safe_load(open(config_filename))
    dst_root_path = config["dst_root_path"]
    interpolated_gt_path = config["interpolated_gt_path"]
    total_save_path = config["total_save_path"] 
    # seq = config["seq"]

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

    # # for seq in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
    # for seq in ["01"]:
    #     print(seq)

    #     abs_interpolated_gt_path = os.path.join(dst_root_path, seq, interpolated_gt_path)
    #     abs_total_save_path = os.path.join(dst_root_path, seq, total_save_path)
    #     abs_valid_pose_save_path = os.path.join(dst_root_path, seq, 'train_sets/valid_poses.csv')

    #     # search_seqs = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    #     search_seqs = ["02", "03"]
    #     abs_interloop_search_paths = []
    #     for search_seq in search_seqs:
    #         abs_interloop_search_path = os.path.join(dst_root_path, search_seq, interpolated_gt_path)
    #         abs_interloop_search_paths.append(abs_interloop_search_path)

    #     # generate_total_pairs_kdtree(abs_interpolated_gt_path, abs_interloop_search_paths, 
    #     #                             seq, search_seqs, abs_total_save_path)
    #     save_valid_poses(abs_interpolated_gt_path, abs_valid_pose_save_path, min_interval=1)

    #     count_samples(abs_interpolated_gt_path, abs_interloop_search_paths, 
    #                                 seq, search_seqs, min_interval=1)
    # # pose_array = read_interpolated_pose(interpolated_gt_path)
    # # for i in range(9580, 9600):
    #     # print(check_loop(pose_array[5300,:], pose_array[i,:]))
    
    min_interval = 1
    print("min interval: ", min_interval)

    for seq1, seq2 in pairs:
        abs_query_gt_path = os.path.join(dst_root_path, seq1, interpolated_gt_path)
        abs_total_save_path = os.path.join(dst_root_path, seq1, total_save_path)
        abs_valid_pose_save_path = os.path.join(dst_root_path, seq1, 'train_sets/valid_poses.csv')
        abs_search_gt_path = os.path.join(dst_root_path, seq2, interpolated_gt_path)
        
        save_valid_poses(abs_query_gt_path, abs_valid_pose_save_path, min_interval=min_interval)
        count_samples(abs_query_gt_path, abs_search_gt_path, 
                                    seq1, seq2, min_interval=min_interval)
        print("")