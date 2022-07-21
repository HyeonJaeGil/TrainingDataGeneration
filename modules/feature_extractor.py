import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
from tools.read_samples import read_one_data_from_seq
import yaml
from torchvision import datasets, transforms, models


class featureExtracter(nn.Module):
    def __init__(self, height=400, width=640, channels=3, norm_layer=None):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        model = models.resnet50(pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        
        #self.newmodel[8] = nn.AdaptiveAvgPool2d((1,1))
        print(self.newmodel)
        # self.newmodel[8] = torch.nn.AvgPool2d()
        self.fc_layers = torch.nn.Sequential(
            nn.Linear(model.fc.in_features, 4 * 128),
            nn.Linear(4 * 128, 2 * 128),
            nn.Linear(2 * 128, 1 * 128)
        )

        # self.l2_norm = torch.nn.Sequential()


        # self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        # self.bn1 = norm_layer(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False)
        # self.bn2 = norm_layer(32)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        # self.bn3 = norm_layer(64)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        # self.bn4 = norm_layer(64)
        # self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False)
        # self.bn5 = norm_layer(128)
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn6 = norm_layer(128)
        # self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn7 = norm_layer(128)
        # self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn8 = norm_layer(128)
        # self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn9 = norm_layer(128)
        # self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn10 = norm_layer(128)
        # self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        # self.bn11 = norm_layer(128)
        # self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        # self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        # self.bnLast1 = norm_layer(256)
        # self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        # self.bnLast2 = norm_layer(1024)

        # self.linear = nn.Linear(128*900, 256)

        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        # self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
        #                              output_dim=256, gating=True, add_batch_norm=False,
        #                              is_training=True)

        """TODO: How about adding some dense layers?"""
        # self.linear1 = nn.Linear(1 * 256, 256)
        # self.bnl1 = norm_layer(256)
        # self.linear2 = nn.Linear(1 * 256, 256)
        # self.bnl2 = norm_layer(256)
        # self.linear3 = nn.Linear(1 * 256, 256)
        # self.bnl3 = norm_layer(256)

    def forward(self, x_l):

        out_l = self.newmodel(x_l)

        # print(out_l.shape, out_l.dtype)
        out_l=out_l.view(-1,1,2048)
        #print(out_l.shape)
        # out_l_permute = out_l.permute(0, 2, 3, 1)
        # print(out_l_permute, out_l_permute.shape, out_l_permute.dtype)
        out_l = self.fc_layers( out_l)
        out_l = F.normalize(out_l)

        # print(out_l.shape)
        return out_l

        # out_l_1 = out_l.permute(0,1,3,2)
        # out_l_1 = self.relu(self.convLast1(out_l_1))

        # out_l = torch.cat((out_l_1, out_l_1), dim=1)
        # out_l = F.normalize(out_l, dim=1)
        # out_l = self.net_vlad(out_l)
        # out_l = F.normalize(out_l, dim=1)

        # return out_l


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/train_config.yaml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root_folder"]
    # ============================================================================

    combined_tensor = read_one_data_from_seq(seqs_root, 103 ,2)
    combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)
    print(combined_tensor.shape)

    feature_extracter=featureExtracter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()

    print("model architecture: \n")
    # print(feature_extracter)

    gloabal_descriptor = feature_extracter(combined_tensor)
    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor.size())
