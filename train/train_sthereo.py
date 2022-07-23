import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../')
import torch
import numpy as np
from tensorboardX import SummaryWriter
# from tools.read_all_sets import overlap_orientation_npz_file2string_string_nparray
from modules.feature_extractor import featureExtracter
# from tools.read_samples import read_one_batch_pos_neg
# from tools.read_samples import read_one_need_from_seq
import modules.loss as loss_module
# from tools.utils.utils import *
# from valid.valid_seq import validate_seq_faiss
from tools.read_samples import csvfiles_to_nparray, read_one_batch_pos_neg, read_one_data
np.set_printoptions(precision=9)
import yaml


class trainHandler():
    def __init__(self, height=400, width=640, channels=1, norm_layer=None, lr=0.001,
                data_root_folder = None, train_sets=None, training_seqs=None):
        
        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.learning_rate = lr
        self.data_root_folder = data_root_folder
        self.train_sets = train_sets
        self.training_seqs = training_seqs
        
        self.amodel = featureExtracter(channels=self.channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.parameters  = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.traindata_csvfiles = train_sets

        (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.label) = \
            csvfiles_to_nparray(self.traindata_csvfiles)

        self.resume = False
        self.save_name = "../weights/pretrained_weight.pth.tar"


    def train(self):

        epochs = 100
        
        """resume from the saved model"""
        if self.resume:
            resume_filename = self.save_name
            print("Resuming from ", resume_filename)
            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ..." )
            starting_epoch = 0
       
        writer1 = SummaryWriter(comment="LR_0.xxxx")

        for i in range(starting_epoch+1, epochs):

            (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.label) = \
                csvfiles_to_nparray(self.traindata_csvfiles, shuffle=True)

            print("=======================================================================\n\n\n")

            print("training with seq: ", np.unique(np.array(self.train_dir1)))
            print("total pairs: ", len(self.train_imgf1))
            print("\n\n\n=======================================================================")

            loss_each_epoch = 0
            used_num = 0

            for j in range(len(self.train_imgf1)):
                f1_index = self.train_imgf1[j]
                dir1_index = self.train_dir1[j]
                print(int(f1_index), int(dir1_index))
                current_batch = read_one_data(self.data_root_folder, f1_index, dir1_index)
                if current_batch.numel() == 1:
                    continue 

                sample_batch, sample_truth, pos_num, neg_num = read_one_batch_pos_neg \
                    (self.data_root_folder,f1_index, dir1_index,
                        self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.label)

                use_pos_num = 6
                use_neg_num = 6
                if pos_num >= use_pos_num and neg_num>=use_neg_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:pos_num+use_neg_num, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:pos_num+use_neg_num, :]), dim=0)
                    pos_num = use_pos_num
                    neg_num = use_neg_num
                elif pos_num >= use_pos_num:
                    sample_batch = torch.cat((sample_batch[0:use_pos_num, :, :, :], sample_batch[pos_num:, :, :, :]), dim=0)
                    sample_truth = torch.cat((sample_truth[0:use_pos_num, :], sample_truth[pos_num:, :]), dim=0)
                    pos_num = use_pos_num
                elif neg_num >= use_neg_num:
                    sample_batch = sample_batch[0:pos_num+use_neg_num,:,:,:]
                    sample_truth = sample_truth[0:pos_num+use_neg_num, :]
                    neg_num = use_neg_num

                if neg_num == 0:
                    continue

                input_batch = torch.cat((current_batch, sample_batch), dim=0)
                print(input_batch.shape)

                input_batch.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad()

                global_des = self.amodel(input_batch)
                o1, o2, o3 = torch.split(
                    global_des, [1, pos_num, neg_num], dim=0)
                MARGIN_1 = 0.5
                # print(o1 , o1.shape, o1.dtype)
                # print(o2, o2.shape, o2.dtype)

                loss = loss_module.triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)
                # loss = loss.triplet_loss_inv(o1, o2, o3, MARGIN_1, lazy=False, use_min=True)
                loss.backward()
                self.optimizer.step()
                print(str(used_num), loss)

                if torch.isnan(loss):
                    print("Something error ...")
                    print(pos_num)
                    print(neg_num)

                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

            print("epoch {} loss {}".format(i, loss_each_epoch/used_num))
            print("saving weights ...")
            self.scheduler.step()

            """save trained weights and optimizer states"""
            self.save_name = "../weights/pretrained_weight"+str(i)+".pth.tar"

            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                self.save_name)

            print("Model Saved As " + 'pretrained_weight' + str(i) + '.pth.tar')

            writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)
         
            print("validating ......")
            with torch.no_grad():
                # do Validation
                pass


if __name__ == '__main__':
    config_filename = '../config/train_config.yaml'
    config = yaml.safe_load(open(config_filename))
    data_root_folder = config["data_root_folder"]
    training_seqs = config["training_seqs"]
    training_seqs = ["07"]

    training_datas = [os.path.join(data_root_folder, seq, 'train_sets/train_pairs.csv') for seq in training_seqs]

    train_handler = trainHandler(height=400, width=640, channels=3, norm_layer=None, lr=0.000005,
                                 data_root_folder=data_root_folder, train_sets=training_datas, training_seqs = training_seqs)

    train_handler.train()