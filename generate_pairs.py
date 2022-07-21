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


def check_soft_negative(np_1, np_2, min_dist=6, max_dist=20):

        distance = ((np_1 - np_2) ** 2).sum(0)
        if distance < max_dist**2 and distance > min_dist**2:
            return True
        else:
            return False


def check_hard_negative(np_1, np_2, min_dist=50):

        distance = ((np_1 - np_2) ** 2).sum(0)
        if distance > min_dist**2:
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


def soft_negative_sample_search(np_query, np_db, min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    for np_pose in np_db:
        is_soft_neg = check_soft_negative(np_query[1:4], np_pose[1:4])
        if is_soft_neg:
            ids.append(int(np_pose[0]))
    return np.array(ids)


def soft_negative_sample_search_with_valid_ids(np_query, np_db, valid_ids, 
                min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    # for np_pose in np_db:
    for valid_id in valid_ids:
        is_soft_neg = check_soft_negative(np_query[1:4], np_db[valid_id,1:4])
        if is_soft_neg:
            ids.append(int(valid_id))
    return np.array(ids)


def hard_negative_sample_search_with_valid_ids(np_query, np_db, valid_ids, 
                min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    ids = []
    # for np_pose in np_db:
    for valid_id in valid_ids:
        is_soft_neg = check_hard_negative(np_query[1:4], np_db[valid_id, 1:4])
        if is_soft_neg:
            ids.append(int(valid_id))
    return np.array(ids)


def candidates_search_with_valid_ids(np_query, np_db, valid_ids, 
                min_interval=300, max_dist=5, heading_tolerance = 3000):
 
    loop_ids = []
    softneg_ids = []
    hardneg_ids = []
    # for np_pose in np_db:
    for valid_id in valid_ids:
        is_loop = check_loop(np_query, np_db[valid_id, :])
        if is_loop:
            loop_ids.append(int(valid_id))
       
        is_soft_neg = check_soft_negative(np_query[1:4], np_db[valid_id, 1:4])
        if is_soft_neg:
            softneg_ids.append(int(valid_id))
       
        is_hard_neg = check_hard_negative(np_query[1:4], np_db[valid_id, 1:4])
        if is_hard_neg:
            hardneg_ids.append(int(valid_id))

    return np.array(loop_ids), np.array(softneg_ids), np.array(hardneg_ids)


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


def generate_filtered_pose_array(query_gt_path):
    
    pose_array1 = read_interpolated_pose(query_gt_path)
    valid_poses_ids1 = generate_valid_poses(query_gt_path)

    filtered_array = np.array([], dtype="float").reshape(0,pose_array1.shape[-1])
    for id in valid_poses_ids1:
        filtered_array = np.vstack([filtered_array, pose_array1[id, :]])

    # print(filtered_array, filtered_array.shape)
    return filtered_array


def generate_inter_loop_pairs(query_gt_path, search_gt_paths, seq1, seqs, output_path, max_dist=3):
    
    # pose_array1 = read_interpolated_pose(query_gt_path)
    # valid_poses_ids1 = generate_valid_poses(query_gt_path)

    valid_query_pose_array = generate_filtered_pose_array(query_gt_path)
    # pose_array2 = read_interpolated_pose(gt_pose_path2)
    # valid_poses_ids2 = generate_valid_poses(gt_pose_path2)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)

        for i in range(len(search_gt_paths)):
            count = 0
            search_seq = seqs[i]
            # pose_array2 = read_interpolated_pose(search_gt_paths[i])
            # valid_poses_ids2 = generate_valid_poses(search_gt_paths[i])

            valid_search_pose_array = np.transpose(generate_filtered_pose_array(search_gt_paths[i]), (1, 0))
            
            print(valid_query_pose_array.shape, valid_search_pose_array.shape)

            rep_valid_query_pose_array = np.repeat \
                (valid_query_pose_array[:,:,np.newaxis], valid_search_pose_array.shape[1], axis=2)
            # rep_valid_search_pose_array = np.repeat \
            #     (valid_search_pose_array[:,:,np.newaxis], valid_query_pose_array.shape[0], axis=2)

            # print(rep_valid_query_pose_array.shape, rep_valid_search_pose_array.shape)

            # rep_valid_search_pose_array = np.transpose(rep_valid_search_pose_array, [2,1,0])       

            # distance_3d_array = rep_valid_query_pose_array - rep_valid_search_pose_array
            distance_3d_array = rep_valid_query_pose_array - valid_search_pose_array
            print(distance_3d_array.shape)


            for i in range(distance_3d_array.shape[0]):
                for j in range(distance_3d_array.shape[2]):
                    dist_ij = distance_3d_array[i, :, j]
                    if (dist_ij[1:4]**2).sum() < 25 and (dist_ij[4:]**2).sum() < 900 and abs(dist_ij[0]) > 300:
                        writer.writerow([i+1, j+1, seq1, search_seq, 1])
                    elif (dist_ij[1:4]**2).sum() > 36 and (dist_ij[1:4]**2).sum() < 400:
                        writer.writerow([i+1, j+1, seq1, search_seq, 0.5])
                    elif (dist_ij[1:4]**2).sum() > 2500:
                        writer.writerow([i+1, j+1, seq1, search_seq, 0.0])
                    count += 1
            print('finished seq number: ', i)
            print('total loop count: ', count)

            # for pose_id in valid_poses_ids1:
            #     query = pose_array1[pose_id-1, :]
            #     # candidates = positive_sample_search(query, pose_array2)
            #     candidates = positive_sample_search_with_valid_ids(query, pose_array2, valid_poses_ids2)
            #     for candidate in candidates:
            #         writer.writerow([query[0], candidate, seq1, search_seq, 1])
            #         count += 1


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


def generate_inter_soft_neg_pairs(query_gt_path, search_gt_paths, seq1, seqs, output_path, max_dist=3):
    
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
                candidates = soft_negative_sample_search_with_valid_ids(query, pose_array2, valid_poses_ids2)
                for candidate in candidates:
                    writer.writerow([query[0], candidate, str(seq1), str(search_seq), 0.5])
                    count += 1
            print('finished seq number: ', i)
            print('total soft neg count: ', count)


def generate_inter_hard_neg_pairs(query_gt_path, search_gt_paths, seq1, seqs, output_path, max_dist=3):
    
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
                candidates = hard_negative_sample_search_with_valid_ids(query, pose_array2, valid_poses_ids2)
                for candidate in candidates:
                    writer.writerow([query[0], candidate, seq1, search_seq, 0.0])
                    count += 1
            print('finished seq number: ', i)
            print('total hard neg count: ', count)


def generate_total_pairs(query_gt_path, search_gt_paths, seq1, seqs, output_path, 
                    p_loop=0.3, p_softneg=0.3, p_hardneg=0.1, max_dist=3):
    
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
                loop_candidates, softneg_candidates, hardneg_candidates = \
                candidates_search_with_valid_ids(query, pose_array2, valid_poses_ids2)
                
                for candidate in loop_candidates:
                    writer.writerow([query[0], candidate, seq1, search_seq, 1.0])
                    count += 1
                for candidate in softneg_candidates:
                    writer.writerow([query[0], candidate, seq1, search_seq, 0.5])
                    count += 1
                for candidate in hardneg_candidates:
                    writer.writerow([query[0], candidate, seq1, search_seq, 0.0])
                    count += 1                    
           
            print('finished seq number: ', seqs[i])
            print('total count: ', count)


'''
kaist: 0.5, 0.4, 0.2, 0.2, 50, 20, 10 
snu: 0.7, 0.2, 0.1
valley: 0.5, 0.3, 0.1
'''

def generate_total_pairs_kdtree(query_gt_path, search_gt_paths, seq1, seqs, output_path, hardneg_min_dist=50,
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
            search_seq = seqs[i]
            s_pose_array = generate_filtered_pose_array(search_gt_paths[i])
            s_index_array = s_pose_array[:, 0]
            s_position_array = s_pose_array[:, 1:4]
            s_heading_array = s_pose_array[:, 4:]
            
            pose_tree = KDTree(s_position_array)

            for q_idx in range(q_position_array.shape[0]):
                pos_samples, softneg_samples, hardneg_samples = 0, 0, 0
                
                if check_true(p_loop):
                    for idx in pose_tree.query_ball_point(q_position_array[q_idx, :], r=pos_max_dist, p=2):
                    # for idx in idxs:
                        if check_true(p_dropout) and \
                        ((q_heading_array[q_idx,:]-s_heading_array[idx,:])**2).sum() < max_heading_diff**2 and \
                        abs(q_index_array[q_idx]-s_index_array[idx]) > min_id_diff:
                            writer.writerow([q_index_array[q_idx],s_index_array[idx], seq1, search_seq, 1])
                            pos_samples += 1
                            count += 1
                        if pos_samples >= max_pos_samples: break
                # if check_true(p_softneg) and pos_samples > 0:
                if pos_samples > 0:
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
               
                # if check_true(p_hardneg) and pos_samples > 0:
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
    # seq = config["seq"]
    interpolated_gt_path = config["interpolated_gt_path"]
    intraloop_save_path = config["intraloop_save_path"]
    interloop_save_path = config["interloop_save_path"] 
    inter_softneg_save_path = config["inter_softneg_save_path"] 
    inter_hardneg_save_path = config["inter_hardneg_save_path"] 
    total_save_path = config["total_save_path"] 

# ["01", "02", "03"]
# ["04", "05", "06"]
# ["07", "08", "09"]

    # for seq in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
    for seq in ["01"]:
        print(seq)

        abs_interpolated_gt_path = os.path.join(dst_root_path, seq, interpolated_gt_path)
        abs_intraloop_save_path = os.path.join(dst_root_path, seq, intraloop_save_path)
        abs_interloop_save_path = os.path.join(dst_root_path, seq, interloop_save_path)
        abs_inter_softneg_save_path = os.path.join(dst_root_path, seq, inter_softneg_save_path)
        abs_inter_hardneg_save_path = os.path.join(dst_root_path, seq, inter_hardneg_save_path)
        abs_total_save_path = os.path.join(dst_root_path, seq, total_save_path)

        # valid_pose = generate_valid_poses(abs_interpolated_gt_path)
        # print(valid_pose.shape)
        # filtered_array = generate_filtered_pose_array(abs_interpolated_gt_path)
        # print(filtered_array.shape)
        # generate_intra_loop_pairs(abs_interpolated_gt_path, abs_intraloop_save_path, seq=seq)

        # search_seqs = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
        search_seqs = ["01", "02", "03"]
        abs_interloop_search_paths = []
        for search_seq in search_seqs:
            abs_interloop_search_path = os.path.join(dst_root_path, search_seq, interpolated_gt_path)
            abs_interloop_search_paths.append(abs_interloop_search_path)
        
        # generate_inter_loop_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
        #                             seq, search_seqs, abs_interloop_save_path)
        # generate_inter_loop_pairs_kdtree(abs_interpolated_gt_path, abs_interloop_search_paths, 
        #                             seq, search_seqs, abs_interloop_save_path)

        # generate_inter_soft_neg_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
        #                             seq, search_seqs, abs_inter_softneg_save_path)            
        # generate_inter_hard_neg_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
                                    # seq, search_seqs, abs_inter_hardneg_save_path)
        # generate_total_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
        #                             seq, search_seqs, abs_total_save_path)
        generate_total_pairs_kdtree(abs_interpolated_gt_path, abs_interloop_search_paths, 
                                    seq, search_seqs, abs_total_save_path)
    # generate_inter_loop_pairs(abs_interpolated_gt_path, abs_interloop_search_paths, 
    #                             seq, search_seqs, abs_interloop_save_path)

    # pose_array = read_interpolated_pose(interpolated_gt_path)
    # for i in range(9580, 9600):
        # print(check_loop(pose_array[5300,:], pose_array[i,:]))
    
