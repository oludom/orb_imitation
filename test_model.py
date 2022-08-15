#!/usr/bin/env python

'''

simulation client for AirSimInterface.py
this runs the main loop and holds the settings for the simulation. 


'''

from email import parser
import sys
from urllib import response
import os



# # os.chdir('./datagen')

# sys.path.append('../')
# sys.path.append('../../')
# sys.path.append('../../../')


import numpy as np
import pprint
import curses
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import os
import time
from math import *
import time

import cv2
from copy import deepcopy
from path import Path
# os.chdir('../')
# print(os.getcwd())
# import imitation.ResNet8 as resnet8
# import models.racenet8 as racenet8
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))
from models.racenet8 import RaceNet8
from models.ResNet8 import ResNet8



torch.set_grad_enabled(False)

parser = argparse.ArgumentParser('Add argument for AirsimClient')
parser.add_argument('--weight','-w',type=str,default='')
parser.add_argument('--architecture','-arc', type=str, choices=["resnet8", "racenet8"])
arg = parser.parse_args()
arc = arg.architecture
model_weight_path = arg.weight
images_dir = Path("/media/data2/teamICRA/X4Gates_Circles_rl18tracks/Onegate-l-3/track1/image_left")
device = "cpu"
if arc == 'resnet8':
    model = ResNet8(input_dim=3, output_dim=4, f=.5)
if arc == 'racenet8':
    model = RaceNet8(input_dim=3, output_dim=4, f=.5)
        
# if device == 'cuda':
#     self.model = nn.DataParallel(self.model)
#     cudnn.benchmark = True
model.load_state_dict(torch.load(model_weight_path))

device = device
dev = torch.device(device)
model.to(dev)
model.eval()

for img in os.listdir(images_dir):
    image_path = images_dir + '/' + Path(img)
    image = cv2.imread(image_path)
    # print(type(image))
    image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)
    image = torch.unsqueeze(image, dim= 0)
    pred = model(image)
    pred = pred.to(torch.device('cpu'))
    pred = pred.detach().numpy()
    pred = pred[0]
    print(pred)
        