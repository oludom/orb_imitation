

import sys

from sklearn.utils import shuffle
sys.path.insert(0, './datagen')


from  SimClient import SimClient
from DaggerClient import DaggerClient
from train import *    # also imports config

import numpy as np




# with contextlib.closing(SimClient(configFilePath='datagen/config.json')) as dc:
#     # dc.printGatePositions(8)
#     gatePositions = dc.config.gates['poses']
#     dc.gateConfigurations = [gatePositions]
#     dc.gateMission(True, False)



from operator import mod
from pyexpat import model
from re import I
from signal import pause
from statistics import mode
from tokenize import Triple
from turtle import shape
from SimClient import SimClient
from NetworkTestClient import NetworkTestClient
import time
import numpy as np
from UnityPID import VelocityPID
from copy import deepcopy
import airsim
import torch
import argparse
from AirSimInterface import AirSimInterface
from util import *
from math import *
import os
from ResNet8 import ResNet8
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import curses

# from train_newloader import train_main


if __name__ == "__main__":

    import contextlib

    beta = 1

# tb path handling
    TB_path, writer = get_path_for_run(config.project_basepath, config.dataset_basename, config.itypes, config.resnet_factor, config.batch_size, config.loss_type, config.learning_rate, config.TB_suffix)

    # path, writer = get_path_for_run(config.project_basepath, config.dataset_basename, config.itypes, config.resnet_factor, config.batch_size, config.loss_type, config.learning_rate, config.TB_suffix)
    # model_path = str(TB_path) + f"epoch{i-1}.pth"

    dev = torch.device(config.device)

    model = ResNet8(input_dim=config.num_input_channels, output_dim=4, f=config.resnet_factor)
    model = model.to(dev)


    if config.parallel and config.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    lossfunction = nn.MSELoss()

    # load initial model 
    if config.initial_weight_path:
        model.load_state_dict(torch.load(config.initial_weight_path))

    configurations = []

    with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_cleft.json', createDataset=False)) as dc:
        # generate random gate configurations within bounds set in config.json
        dc.generateGateConfigurations()
        configurations1 = deepcopy(dc.gateConfigurations)

    with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_cright.json', createDataset=False)) as dc:
        # generate random gate configurations within bounds set in config.json
        dc.generateGateConfigurations()
        configurations2 = deepcopy(dc.gateConfigurations)

    configurations = list(configurations1) + list(configurations2)

    configurations = shuffle(configurations)


    step_pos = {}
    for el in config.phases:
        step_pos[el] = 0

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    try:
        # for epoch in range(config.epochs):
            # train epoch
        # for each configuration
        for i, gateConfig in enumerate(configurations):

            with contextlib.closing(
                DaggerClient(model, beta=beta, device=config.device, raceTrackName=f"track{i+config.skip_tracks}", configFilePath='datagen/config.json', createDataset=True)) as dc:
                dc.gateConfigurations = [gateConfig]

                # load next gate arrangement 
                dc.loadNextGatePosition()

                # fly mission
                dc.run(showMarkers=False, uav_position=dc.config.uav_position)

            datasets = load_dataset_train_val_split(config.dataset_basepath, config.dataset_basename, config.device, 1000,
                                config.input_channels, config.skipFirstXImages, config.skipLastXImages, config.batch_size, config.tf, config.jobs)

            batch_count = {}
            for el in config.phases:
                batch_count[el] = len(datasets[el])
                print(f"batch count {el}: {batch_count[el]}")
                                
            model = train_epoch(i, len(configurations), model, config.phases, config.learning_rate, config.learning_rate_change, config.learning_rate_change_epoch, datasets, dev, lossfunction, writer, step_pos, TB_path, batch_count)
            
            beta = 1 / (i+2)

    except Exception as e:
        raise e
    finally:
        # save model
        writer.flush()
        writer.close()

