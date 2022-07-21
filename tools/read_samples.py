import torch
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
import yaml
from utils import read_pairs, read_image, read_image_16bit
import csv
from preprocess_images import to2digit, to6digit
np.set_printoptions(precision=9)


def check_total_batch(train_csvfiles):

    (train_imgf1, train_imgf2, train_dir1, train_dir2, train_loop) = \
                csvfiles_to_nparray(train_csvfiles, False)   
    used_list_f1 = []
    used_list_dir1 = []

    for j in range(len(train_imgf1)):
        """
            check whether the query is used to train before (continue_flag==True/False).
            TODO: More efficient method
        """
        f1_index = train_imgf1[j]
        dir1_index = train_dir1[j]

        first_time = True
        for iddd in range(len(used_list_f1)):
            if f1_index==used_list_f1[iddd] and dir1_index==used_list_dir1[iddd]:
                first_time = False

        if first_time:
            used_list_f1.append(f1_index)
            used_list_dir1.append(dir1_index)
            print("append ", f1_index, dir1_index)

    print(len(used_list_f1), len(used_list_dir1))

    return used_list_f1, used_list_dir1


def csvfiles_to_nparray(csvfile_list, shuffle=True):
    imgf1_all = []
    imgf2_all = []
    dir1_all = []
    dir2_all = []
    loop_all = []

    for csvfile in csvfile_list:
        print(csvfile)
        pairs_array = read_pairs(csvfile)
        imgf1 = pairs_array[:, 0].tolist()                
        imgf2 = pairs_array[:, 1].tolist()                
        dir1 = pairs_array[:, 2].tolist()                
        dir2 = pairs_array[:, 3].tolist()                
        loop = pairs_array[:, 4].tolist()                

        # seq, overlaps
        # imgf1 = np.char.mod('%06d', h['overlaps'][:, 0]).tolist()
        # imgf2 = np.char.mod('%06d', h['overlaps'][:, 1]).tolist()
        # overlap = h['overlaps'][:, 2]
        # dir1 = (h['seq'][:, 0]).tolist()
        # dir2 = (h['seq'][:, 1]).tolist()

        if shuffle:
            shuffled_idx = np.random.permutation(len(loop))
            print(shuffled_idx)
            imgf1 = (np.array(imgf1)[shuffled_idx]).tolist()
            imgf2 = (np.array(imgf2)[shuffled_idx]).tolist()
            dir1 = (np.array(dir1)[shuffled_idx]).tolist()
            dir2 = (np.array(dir2)[shuffled_idx]).tolist()
            loop = (np.array(loop)[shuffled_idx]).tolist()

        imgf1_all.extend(imgf1)
        imgf2_all.extend(imgf2)
        dir1_all.extend(dir1)
        dir2_all.extend(dir2)
        loop_all.extend(loop)

    return (imgf1_all, imgf2_all, dir1_all, dir2_all, np.asarray(loop_all))



def read_one_data_from_seq(data_root_folder, file_num, seq_num):
    # print(os.path.join(data_root_folder, to2digit(seq_num), "thermal_left", to6digit(file_num))+'.png')
    # img_data = np.array(cv2.imread(
    #     os.path.join(data_root_folder, to2digit(seq_num), "thermal_left", to6digit(file_num)+'.png'), 0))

    img_data = read_image(
        os.path.join(data_root_folder, to2digit(seq_num), "thermal_left", to6digit(file_num)+'.png'))

    if img_data.size == 0:
        print("empty data: producing empty tensor.")
        return torch.zeros(1)

    # if img_data == -1:
    #     print("invalid image")

    # print(img_data, img_data.shape, img_data.dtype)
    img_data = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)
    img_data_tensor = torch.from_numpy(img_data).type(torch.FloatTensor).cuda()
    img_data_tensor = torch.unsqueeze(img_data_tensor, dim=0)
    # img_data_tensor = torch.unsqueeze(img_data_tensor, dim=0)
    # img_data_tensor = np.transpose(img_data_tensor,(0, 2, 3, 1))

    img_data_tensor = img_data_tensor.reshape(-1, 3, 400, 640)

    return img_data_tensor


"""
    read one batch of positive samples and negative samples with respect to $f1_index in sequence $f1_seq.
    Args:
        data_root_folder: dataset root of STHEREO.
        f1_index: the index of the needed scan (zfill 6).
        f1_seq: the sequence in which the needed scan is (zfill 2).
        train_imgf1, train_imgf2, train_dir1, train_dir2: the index dictionary and sequence dictionary following OverlapNet.
        train_overlap: overlaps dictionary following OverlapNet.
        overlap_thresh: 0.3 following OverlapNet.
"""
def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, 
                            train_dir1, train_dir2, train_label):  # without end

    threshold = 0.51

    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 3, 400, 640))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_label[j]> threshold:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1

            filename = os.path.join(
                data_root_folder, to2digit(train_dir2[j]), "thermal_left", to6digit(train_imgf2[j])) + ".png"
            # print(filename)

            if not os.path.isfile(filename):
                continue

            img_data_r_cv = read_image(filename)
            img_data_r_cv = cv2.applyColorMap(img_data_r_cv, cv2.COLORMAP_JET)
            


            img_data_r = np.array(img_data_r_cv)
            img_data_r.reshape(3, 400, 640)

            img_data_tensor_r = torch.from_numpy(img_data_r).type(torch.FloatTensor).cuda()
            img_data_tensor_r = torch.unsqueeze(img_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = img_data_tensor_r.reshape(-1, 3, 400, 640)
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_label[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = img_data_tensor_r.reshape(-1, 3,400, 640)
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_label[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1


    return sample_batch, sample_truth, pos_num, neg_num



if __name__ == '__main__':
    # img = read_one_data_from_seq("/media/hj/seagate/datasets/STHEREO", "1630106835306123587", "01")
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    config_filename = '../config/train_config.yaml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root_folder"]
    training_seqs = config["training_seqs"]

    # csv_files = \
    # ['/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/01/train_sets/interloop.csv',
    # '/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/02/train_sets/interloop.csv',
    # '/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/03/train_sets/interloop.csv',
    # '/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/01/train_sets/softneg.csv',
    # '/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/02/train_sets/softneg.csv',
    # '/media/hj/seagate/datasets/sthereo_datasets/STHEREO/dataset_full/03/train_sets/softneg.csv'
    # ]

    # (train_imgf1, train_imgf2, train_dir1, train_dir2, train_loop) = \
    # csvfiles_to_nparray(csv_files, False)

    # with open('/home/hj/csvtest.csv', 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(train_imgf1)):
    #         writer.writerow([train_imgf1[i], train_imgf2[i], train_dir1[i], train_dir2[i], train_loop[i]])

    training_seqs = ["07"]
    train_csvfiles = [os.path.join(data_root_folder, to2digit(seq), 'train_sets/train_pairs.csv') for seq in training_seqs]
    (train_imgf1, train_imgf2, train_dir1, train_dir2, train_loop) = \
    csvfiles_to_nparray(train_csvfiles, False)   
    # with open('/home/hj/csvtest.csv', 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(train_imgf1)):
    #         writer.writerow([train_imgf1[i], train_imgf2[i], train_dir1[i], train_dir2[i], train_loop[i]])
    
    # check_total_batch(train_csvfiles)


    sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg \
        (data_root_folder, 17, 7, train_imgf1, train_imgf2, train_dir1, train_dir2, train_loop)

    # print(sample_batch.shape, sample_truth.shape, pos_num, neg_num)