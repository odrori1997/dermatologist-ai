#!/usr/bin/env python
# coding: utf-8

## define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 224-5/1 + 1 = 220
        # output tensor: (32, 220, 220)
        # after maxpool: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        
        # 110-3/1 + 1 = 108
        # output tensor: (64, 108, 108)
        # after maxpool: (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        
        self.pool = nn.MaxPool2d(2,2)
        self.drop_layer1 = nn.Dropout(p=0.4)
        self.drop_layer2 = nn.Dropout(p=0.2)
        
        
        self.lin1 = nn.Linear(64*54*54, 3)
        
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop_layer1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop_layer2(x)
        
        # prepare for linear layer by flattening
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        
        return x

