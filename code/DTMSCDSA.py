import torch
from torch import nn
from torch_geometric.nn import GCNConv
torch.backends.cudnn.enabled = False
import csv
import numpy as np
import pandas as pd
import torch.nn.functional as F

from hypergraph_utils import *

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class MSP(nn.Module):
    def __init__(self, channels, factor=5):
        super(MSP, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

class DTMSCDSA(nn.Module):
    def __init__(self, args):
        super(DTMSCDSA, self).__init__()
        self.args = args
        self.gcn_x1_cfs = GCNConv(self.args.fc, self.args.fc)  
        self.gcn_x2_cfs = GCNConv(self.args.fc, self.args.fc)  
        self.gcn_y1_dss = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_dss = GCNConv(self.args.fd, self.args.fd) 
        self.rd_y = nn.Linear(in_features=self.args.circRNA_number,
                               out_features=self.args.out_channels)
        self.rd_x = nn.Linear(in_features=self.args.drug_number,
                               out_features=self.args.out_channels)
        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()
        self.temp=0

        self.cnn_x = nn.Conv2d(in_channels=self.args.circRNA_view*(self.args.gcn_layers+1),
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fc, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv2d(in_channels=self.args.drug_view * (self.args.gcn_layers+1),
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)
        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              padding=7 // 2, bias=False)

    def forward(self, data):
        torch.manual_seed(1)
        dataset_train = dict()
        c_f = []
        d_f = []
        x_c = self.rd_x(data['cd_p'])
        x_d = self.rd_y(data['cd_p'].t())
        d_f.append(x_d.unsqueeze(0))
        c_f.append(x_c.unsqueeze(0))
        
        CKS = construct_H_with_KNN(data['css']['data_matrix'])
        CKS = torch.FloatTensor(CKS)
        CKS_edge = get_edge_index(CKS)
        dataset_train['CKS'] = {'data_matrix': CKS, 'edges': CKS_edge}
        x_c_cfs1 = torch.relu(self.gcn_x1_cfs(x_c, dataset_train['CKS']['edges'], dataset_train['CKS']['data_matrix'][dataset_train['CKS']['edges'][0], dataset_train['CKS']['edges'][1]]))
        cf1 = x_c_cfs1.unsqueeze(0)
        c_f.append(cf1)
        cf1 = cf1.detach().numpy()
        CKS1 = construct_H_with_KNN(cf1)
        CKS1 = torch.FloatTensor(CKS1)
        CKS1_edge = get_edge_index(CKS1)
        dataset_train['CKS1'] = {'data_matrix': CKS1, 'edges': CKS1_edge}
        x_c_cfs2 = torch.relu(self.gcn_x2_cfs(x_c_cfs1, dataset_train['CKS1']['edges'], dataset_train['CKS1']['data_matrix'][dataset_train['CKS1']['edges'][0], dataset_train['CKS1']['edges'][1]]))
        cf2 = x_c_cfs2.unsqueeze(0)
        c_f.append(cf2)
        cf2 = cf2.detach().numpy()
        CKS2 = construct_H_with_KNN(cf2)
        CKS2 = torch.FloatTensor(CKS2)
        CKS2_edge = get_edge_index(CKS2)
        dataset_train['CKS2'] = {'data_matrix': CKS2, 'edges': CKS2_edge}
        x_c_cfs3 = torch.relu(self.gcn_x2_cfs(x_c_cfs2, dataset_train['CKS2']['edges'], dataset_train['CKS2']['data_matrix'][dataset_train['CKS2']['edges'][0], dataset_train['CKS2']['edges'][1]]))
        cf3 = x_c_cfs3.unsqueeze(0)
        c_f.append(cf3)
        cf3 = cf3.detach().numpy()
        CKS3 = construct_H_with_KNN(cf3)
        CKS3 = torch.FloatTensor(CKS3)
        CKS3_edge = get_edge_index(CKS3)
        dataset_train['CKS3'] = {'data_matrix': CKS3, 'edges': CKS3_edge}
        x_c_cfs4 = torch.relu(self.gcn_x2_cfs(x_c_cfs3, dataset_train['CKS3']['edges'], dataset_train['CKS3']['data_matrix'][dataset_train['CKS3']['edges'][0], dataset_train['CKS3']['edges'][1]]))
        cf4 = x_c_cfs4.unsqueeze(0)
        c_f.append(cf4)
        c_f = torch.cat(c_f, dim=0)
        c_f = c_f.unsqueeze_(0)
        block = MSP(5)
        output_c = block(c_f)
        DKS = construct_H_with_KNN(data['dss']['data_matrix'])
        DKS = torch.FloatTensor(DKS)
        DKS_edge =  get_edge_index(DKS)
        dataset_train['DKS'] = {'data_matrix': DKS, 'edges': DKS_edge}
        y_d_dss1 = torch.relu(self.gcn_y1_dss(x_d, dataset_train['DKS']['edges'], dataset_train['DKS']['data_matrix'][dataset_train['DKS']['edges'][0], dataset_train['DKS']['edges'][1]]))
        df1 = y_d_dss1.unsqueeze(0)
        d_f.append(df1)
        df1 = df1.detach().numpy()
        DKS1 = construct_H_with_KNN(df1)
        DKS1 = torch.FloatTensor(DKS1)
        DKS1_edge = get_edge_index(DKS1)
        dataset_train['DKS1'] = {'data_matrix': DKS1, 'edges': DKS1_edge}
        y_d_dss2 = torch.relu(self.gcn_y2_dss(y_d_dss1, dataset_train['DKS1']['edges'], dataset_train['DKS1']['data_matrix'][dataset_train['DKS1']['edges'][0], dataset_train['DKS1']['edges'][1]]))
        df2 = y_d_dss2.unsqueeze(0)
        d_f.append(df2)
        df2 = df2.detach().numpy()
        DKS2 = construct_H_with_KNN(df2)
        DKS2 = torch.FloatTensor(DKS2)
        DKS2_edge = get_edge_index(DKS2)
        dataset_train['DKS2'] = {'data_matrix': DKS2, 'edges': DKS2_edge}
        y_d_dss3 = torch.relu(self.gcn_y2_dss(y_d_dss2, dataset_train['DKS2']['edges'], dataset_train['DKS2']['data_matrix'][dataset_train['DKS2']['edges'][0], dataset_train['DKS2']['edges'][1]]))
        df3 = y_d_dss3.unsqueeze(0)
        d_f.append(df3)
        df3 = df3.detach().numpy()
        DKS3 = construct_H_with_KNN(df3)
        DKS3 = torch.FloatTensor(DKS3)
        DKS3_edge = get_edge_index(DKS3)
        dataset_train['DKS3'] = {'data_matrix': DKS3, 'edges': DKS3_edge}
        y_d_dss4 = torch.relu(self.gcn_y2_dss(y_d_dss3, dataset_train['DKS3']['edges'], dataset_train['DKS3']['data_matrix'][dataset_train['DKS3']['edges'][0], dataset_train['DKS3']['edges'][1]]))
        df4 = y_d_dss4.unsqueeze(0)
        d_f.append(df4)
        d_f = torch.cat(d_f, dim=0)
        d_f = d_f.unsqueeze_(0)
        output_d = block(d_f)
        XM = torch.transpose(output_c, dim0=2, dim1=3)
        YM = torch.transpose(output_d, dim0=2, dim1=3)
        x = self.cnn_x(XM)
        x = x.view(self.args.out_channels, self.args.circRNA_number).t()
        y = self.cnn_y(YM)
        y = y.view(self.args.out_channels, self.args.drug_number).t()
        self.temp=self.temp+1
        savex=x.cpu().detach().numpy()
        save_x=pd.DataFrame(savex)
        save_x.to_csv('..\data\circRNAs Feature.csv')
        savey=y.cpu().detach().numpy()
        save_y=pd.DataFrame(savey)
        save_y.to_csv('..\data\drug Feature.csv')
        # obtain the final representation of circRNA and drug
        if self.temp==self.args.epoch:
            return x, y, x.mm(y.t())
        return x, y, x.mm(y.t())


