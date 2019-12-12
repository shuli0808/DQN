#!/usr/bin/env python

import numpy as np
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_in=[4,84,84], conv_channels=[32, 64, 64],
                 conv_kernels=[8, 4, 3], conv_strides=[4, 2, 1], batch_size = 32,
                 n_atoms=51, n_fc = [256], n_out = 6):
        super(Model, self).__init__()
        #paramters from paper
        self.n_in = n_in
        # Number of actions depending on the game
        self.n_out = n_out
        self.n_atoms = n_atoms
        self.dist_list = []
        self.batch_size = batch_size

        self.Vmin = -10.0
        self.Vmax = 10.0
        self.dz = (self.Vmax - self.Vmin) / float(self.n_atoms - 1)
        self.z = torch.arange(self.Vmin, self.Vmax + self.dz, self.dz)


        c0 = n_in[0]
        h0 = n_in[1]

        self.conv_layers = []
        for c, k, s in zip(conv_channels, conv_kernels, conv_strides):
            # append nn.Conv2d with kernel size k, stride s
            self.conv_layers.append(nn.Conv2d(c0, c, kernel_size = k, stride = s))
            # append nn.ReLU layer
            self.conv_layers.append(nn.ReLU())

            h0 = int(float(h0-k) / s + 1)
            c0 = c
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.fc_layers = []
        h0 = h0 * h0 * conv_channels[-1]
        for i, h in enumerate(n_fc):
            # append Linear and ReLU layers
            self.fc_layers.append(nn.Linear(h0, h))            
            self.fc_layers.append(nn.ReLU())
            h0 = h
        
        self.fc_layers.append(nn.Linear(h, self.n_atoms))
        self.fc_layers.append(nn.Softmax(dim = 1))
        self.fc_layers = nn.Sequential(*self.fc_layers)
        #self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):
        x = x.float() / 256

        # feed x into the self.conv_layers
        x = self.conv_layers(x)
        # (flatten) reshape x into a batch of vectors
        x = x.view(x.size(0), -1)
        # feed x into the self.fc_layers
        dist_list = []
        dist_tensor = torch.zeros((x.size()[0], self.n_out, self.n_atoms))
        # for i in range(self.n_out):
        #     #dist_list.append(self.fc_layers(x).view(self.n_atoms,-1))
        #     dist_tensor[i,:] = self.fc_layers(x) ## [2, 51]
        for i in range(x.size()[0]):
            for j in range(self.n_out):
                dist_tensor[i,j,:] = self.fc_layers(x)[i]
 

        res = torch.zeros((x.size()[0], self.n_out))
        for i in range(x.size()[0]):
            res[i,:] = torch.matmul(dist_tensor[i,:,:], self.z)

        # print("dist len ", len(dist_list))
        # print("dist dimen", dist_list[0].size())
        # print("sum dim = 0 ", torch.sum(dist_list[0], dim = 0))
        # print("sum dim = 1 ", torch.sum(dist_list[0], dim = 1))
        # print("dist_tensor dim ", dist_tensor.size())



        #prob = self.softmax(self.dist_list)


        return dist_tensor, res
