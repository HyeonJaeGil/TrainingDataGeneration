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


def show_loop_images(gt_pose_table, img_folder, id1, id2):

    print(id1, id2)
    img1_filename = gt_pose_table[id1-1, 1]
    img2_filename = gt_pose_table[id2-1, 1]
    img1 = np.array(cv2.imread(os.path.join(img_folder,img1_filename)))
    img2 = np.array(cv2.imread(os.path.join(img_folder,img2_filename)))

    # only for visualization
    imgNormalize1 = cv2.normalize(img1,None,0,255,cv2.NORM_MINMAX)
    imgNormalize2 = cv2.normalize(img2,None,0,255,cv2.NORM_MINMAX)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    img1 = cv2.putText(imgNormalize1, 'id=' + str(id1), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    img2 = cv2.putText(imgNormalize2, 'id=' + str(id2), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    img_pair = np.concatenate((img1, img2), axis=1)

    cv2.imshow('loop', img_pair)
    cv2.waitKey(0) 



def read_files_from_path(file_path):

    for file in os.listdir(file_path):
        data = np.array(cv2.imread(os.path.join(file_path,file)))
        data = os.path.join(file_path,file)
        print(data)


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
                
                new_pose = pose_table[idx, 1:-1] * d2 / (d1+d2) + pose_table[idx+1, 1:-1] * d1 / (d1+d2)

                if pose_table[idx, -1] < 0: 
                    new_yaw_1 = pose_table[idx, -1] + np.float64(360.0)
                else: 
                    new_yaw_1 = pose_table[idx, -1]
                if pose_table[idx+1, -1] < 0: 
                    new_yaw_2 = pose_table[idx+1, -1] + np.float64(360.0)
                else: 
                    new_yaw_2 = pose_table[idx+1, -1]
                interpolated_yaw = new_yaw_1  * d2 / (d1+d2) + new_yaw_2 * d1 / (d1+d2)
                
                if interpolated_yaw > np.float64(180.0):
                    interpolated_yaw -= np.float64(360.0)

                writer.writerow( np.concatenate( ([count], [img], new_pose, [interpolated_yaw]), axis=0))
            # writer.writerow( np.concatenate( ([img[:-4]], new_pose), axis=0))

            count += 1
            # if count > 300:
            #     break
    print(count, "images were processed.")

def visualize_gt_pose(gt_pose_file):
    
    count = 0
    plt.clf()
    x_list = []
    y_list = []
    with open(gt_pose_file, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            count += 1
            if count % 100 == 0:
                x, y, z = row[2:5]
                x_list.append(x)
                y_list.append(y)
            # print(x, y, z)
    np_x_list = np.array(x_list)    
    np_y_list = np.array(y_list)   
    print(np_x_list) 
    plt.plot(np_x_list, np_y_list)
    plt.show()

    # plt.xlim([-6000,600])
    # plt.ylim([-1500,1500])
    plt.legend()
    plt.ion()



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

    local_pose_table = read_pose(os.path.join(abs_pose_folder, 'local_pose.csv'))
    global_pose_table = read_pose(os.path.join(abs_pose_folder, 'global_pose.csv'))
    construct_gt_pose_array(abs_image_folder, local_pose_table, abs_output_local_gt_path)
    construct_gt_pose_array(abs_image_folder, global_pose_table, abs_output_global_gt_path)

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