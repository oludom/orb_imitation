import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


def get_config(name = 'GateNet_rt=30'):

    config = {}
    # Paths to annotation file and source data.
    config['name'] = name
    config['batch_size'] = 32
    config['input_shape'] = (120, 160,3)
    config['output_shape'] = 4
    config['epochs'] = 100
    config['base_learning_rate'] = 0.001
    config['lr_schedule'] = [(0.1, 5), (0.01, 8)]
    config['l2_weight_decay'] = 2e-4
    config['batch_norm_decay'] = 0.997
    config['batch_norm_epsilon'] = 1e-5
    config['optimizer'] = 'adam'
   
    return config

class GateNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        # Define block of GateNet here. 
        # Term related to weight initialization will be added as a function
        # Term related to kernel and bias regularizer with L2 loss will be implemented into the loss function

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(16, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True),   #inplace: modify the input directly to save memory
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(32, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True),   #inplace: modify the input directly to save memory
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(16, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(16, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(16, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # No max pool in this layer but flatten follow
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ),
            nn.BatchNorm2d(16, momentum= config['batch_norm_decay'], eps=config['batch_norm_epsilon']),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()

        self.dense8 = nn.Linear(in_features=5*16*3, out_features=np.prod(config['output_shape']))    # not reshaped

        # Initialize weights
        nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv6[0].weight)


    # # For init weights
    # def weight_init(self):
    #     for block in self._modules:
    #         try:
    #             for m in self._modules[block]:
    #                 nn.init.kaiming_normal_(m)
    #         except:
    #             nn.init.kaiming_normal_(block)


    def forward(self, x):
        # print('input = ',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense8(x)
        x = x.view(self.config['batch_size'], 3, 4, 5) # reshape
        return x

class RaceGateNet(GateNet):
    def __init__(self, config=get_config()):
        super().__init__(config)
        self.dense8 = nn.Linear(in_features=5*16*3, out_features=config['output_shape'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        
        x = self.dense8(x)
        return x