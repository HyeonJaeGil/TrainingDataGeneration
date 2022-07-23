import torch
import cv2
import numpy as np
import os
from os.path import join
import sys
sys.path.append('../')
import yaml
from utils import read_csv, read_training_pairs, read_image, read_image_16bit, to2digit, to6digit
import csv
np.set_printoptions(precision=9)


def check_total_batch(train_csvfiles):

    (img1_ids, img2_ids, seq1, seq2, train_loop) = \
                csvfiles_to_nparray(train_csvfiles, False)   
    used_list_f1 = []
    used_list_seq1 = []

    for j in range(len(img1_ids)):
        """
            check whether the query is used to train before (continue_flag==True/False).
            TODO: More efficient method
        """
        f1_index = img1_ids[j]
        seq1_index = seq1[j]

        first_time = True
        for iddd in range(len(used_list_f1)):
            if f1_index==used_list_f1[iddd] and seq1_index==used_list_seq1[iddd]:
                first_time = False

        if first_time:
            used_list_f1.append(f1_index)
            used_list_seq1.append(seq1_index)
            print("append ", f1_index, seq1_index)

    print(len(used_list_f1), len(used_list_seq1))

    return used_list_f1, used_list_seq1


def csvfiles_to_nparray(csvfile_list, shuffle=True):
    img1_ids_all = []
    img2_ids_all = []
    seq1_all = []
    seq2_all = []
    loop_all = []

    for csvfile in csvfile_list:
        # print(csvfile)
        pairs_array = read_training_pairs(csvfile)
        img1_ids = pairs_array[:, 0].tolist()                
        img2_ids = pairs_array[:, 1].tolist()                
        seq1 = pairs_array[:, 2].tolist()                
        seq2 = pairs_array[:, 3].tolist()                
        loop = pairs_array[:, 4].tolist()                

        if shuffle is True:
            shuffled_idx = np.random.permutation(len(loop))
            # print(shuffled_idx)
            img1_ids = (np.array(img1_ids)[shuffled_idx]).tolist()
            img2_ids = (np.array(img2_ids)[shuffled_idx]).tolist()
            seq1 = (np.array(seq1)[shuffled_idx]).tolist()
            seq2 = (np.array(seq2)[shuffled_idx]).tolist()
            loop = (np.array(loop)[shuffled_idx]).tolist()

        img1_ids_all.extend(img1_ids)
        img2_ids_all.extend(img2_ids)
        seq1_all.extend(seq1)
        seq2_all.extend(seq2)
        loop_all.extend(loop)


    return np.asarray(img1_ids_all, dtype='str'), np.asarray(img2_ids_all), np.asarray(seq1_all), np.asarray(seq2_all), np.asarray(loop_all, dtype='float')
    # return (img1_ids_all, img2_ids_all, seq1_all, seq2_all, np.asarray(loop_all))


def read_one_data(data_root_folder, file_num, seq_num):
    img_data = read_image(
        join(data_root_folder, to2digit(seq_num), "thermal_left", to6digit(file_num)+'.png'))

    if img_data.size == 0:
        print("empty data: producing empty tensor.")
        return torch.zeros(1)

    print(img_data, img_data.shape, img_data.dtype)
    img_data = cv2.applyColorMap(img_data, cv2.COLORMAP_JET) # (400, 640, 3)
    img_data_tensor = torch.from_numpy(np.transpose(img_data, (2,0,1))).type(torch.FloatTensor).cuda() # (3, 400, 640)
    img_data_tensor = torch.unsqueeze(img_data_tensor, dim=0)

    # img_data_tensor = img_data_tensor.reshape(-1, 3, 400, 640)

    return img_data_tensor


def read_one_batch_pos_neg(data_root_folder, anchor_id, anchor_seq, img1_ids, img2_ids, 
                            seq1, seq2, train_label): 

    threshold = 0.51
    anchor_id = to6digit(anchor_id)
    anchor_seq = to2digit(anchor_seq)


    batch_size = 0
    for tt in range(len(img1_ids)):
        if (anchor_id) == (img1_ids[tt]) and (anchor_seq) == to2digit(seq1[tt]):
            batch_size = batch_size + 1
    # print("batch size:" , batch_size)
    sample_batch = torch.from_numpy(np.zeros((batch_size, 3, 400, 640))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(img1_ids)):
        find_pos = False
        if (anchor_id) == img1_ids[j] and anchor_seq==seq1[j]:
            if train_label[j]> threshold:
                pos_num = pos_num + 1
                find_pos = True
            else:
                neg_num = neg_num + 1

            filename = join(data_root_folder, to2digit(seq2[j]), "thermal_left", to6digit(img2_ids[j])) + ".png"
            # print(filename)

            if not os.path.isfile(filename):
                continue

            img = read_image(filename)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            # print(img.dtype, img.shape)
            
            img_np = np.transpose(img,(2,0,1))
            # img_np.reshape(3, 400, 640)

            img_tensor = torch.from_numpy(img_np).type(torch.FloatTensor).cuda()
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

            if find_pos:
                # sample_batch[pos_idx,:,:,:] = img_tensor.reshape(-1, 3, 400, 640)
                sample_batch[pos_idx,:,:,:] = img_tensor
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_label[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                # sample_batch[batch_size-neg_idx-1, :, :, :] = img_tensor.reshape(-1, 3,400, 640)
                sample_batch[batch_size-neg_idx-1, :, :, :] = img_tensor
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_label[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1

    return sample_batch, sample_truth, pos_num, neg_num



if __name__ == '__main__':
    # img = read_one_data("/media/hj/seagate/datasets/STHEREO", "1630106835306123587", "01")
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

    # (img1_ids, img2_ids, seq1, seq2, train_loop) = \
    # csvfiles_to_nparray(csv_files, False)

    # with open('/home/hj/csvtest.csv', 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(img1_ids)):
    #         writer.writerow([img1_ids[i], img2_ids[i], seq1[i], seq2[i], train_loop[i]])

    training_seqs = ["07"]
    train_csvfiles = [os.path.join(data_root_folder, to2digit(seq), 'train_sets/train_pairs.csv') for seq in training_seqs]
    (img1_ids, img2_ids, seq1, seq2, train_loop) = \
    csvfiles_to_nparray(train_csvfiles, False)   
    # with open('/home/hj/csvtest.csv', 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(img1_ids)):
    #         writer.writerow([img1_ids[i], img2_ids[i], seq1[i], seq2[i], train_loop[i]])
    
    # check_total_batch(train_csvfiles)

    print(img1_ids[0])

    sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg \
        (data_root_folder, 1, 7, img1_ids, img2_ids, seq1, seq2, train_loop)

    print(sample_batch.shape, sample_truth.shape, pos_num, neg_num)