from re import I
import sys

from sklearn.utils import shuffle

sys.path.insert(0, './datagen')

from datagen.DaggerClient import DaggerClient
from train import *  # also imports config

from copy import deepcopy
import torch
import math
from GateNet import RaceGateNet
from ResNet8 import ResNet8
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb

if __name__ == "__main__":

    import contextlib

    beta = 1
    weight = 0.1
    beta_rate = 0.52
    def schedular_rate(round):
        return (weight) /((round +1) * beta_rate)
    # tb path handling
    TB_path, writer = get_path_for_run(config.project_basepath, config.dataset_basename, config.itypes,
                                       config.resnet_factor, config.batch_size, config.loss_type, config.learning_rate,
                                       config.TB_suffix)

    dev = torch.device(config.device)

    # model = ResNet8(input_dim=config.num_input_channels, output_dim=4, f=config.resnet_factor)
    model = RaceGateNet()
    model = model.to(dev)

    if config.parallel and config.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    lossfunction = nn.MSELoss()

    # load initial model 
    # if config.initial_weight_path:
    #     model.load_state_dict(torch.load(config.initial_weight_path))

    configurations = []

    # with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_cleft.json', createDataset=False)) as dc:
    #     # generate random gate configurations within bounds set in config.json
    #     dc.generateGateConfigurations()
    #     configurations1 = deepcopy(dc.gateConfigurations)

    # with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_cright.json', createDataset=False)) as dc:
    #     # generate random gate configurations within bounds set in config.json
    #     dc.generateGateConfigurations()
    #     configurations2 = deepcopy(dc.gateConfigurations)

    # configurations = list(configurations1) + list(configurations2)

    # configurations = shuffle(configurations)

    with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_dr.json', createDataset=False)) as dc:
        # generate random gate configurations within bounds set in config.json
        dc.generateGateConfigurations()
        configurations = deepcopy(dc.gateConfigurations)

    step_pos = {}
    for el in config.phases:
        step_pos[el] = 0

    best_model = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    agent_round = 0 
    try:
        # for each configuration
        for i, gateConfig in enumerate(configurations):
            i_contitnue = i 
            wandb.init(project="Resnet16_rgbsparse_new_planner", entity="dungtd2403")
            # wandb.config = {
            #         "learning_rate": 0.001,
            #         "epochs": 2,
            #         "batch_size": 32
            # }
            # use i + 1 if model is not pretrained
            # beta = 1 / (i+2)

            with contextlib.closing(
                DaggerClient(model, beta=beta, device=config.device, raceTrackName=f"track{i+config.skip_tracks}", configFilePath='datagen/config_dr.json', createDataset=True)) as dc:
                dc.gateConfigurations = [gateConfig]

                # load next gate arrangement 
                dc.loadNextGatePosition()

                # fly mission
                dc.run(showMarkers=False, uav_position=dc.config.uav_position)

                bn = dc.config.dataset_basename

            datasets = load_dataset_train_val_split(config.dataset_basepath, bn , config.device, 1000,
                            config.input_channels, config.skipFirstXImages, config.skipLastXImages, config.batch_size, config.tf, config.jobs)

            batch_count = {}
            for el in config.phases:
                batch_count[el] = len(datasets[el])
                print(f"batch count {el}: {batch_count[el]}")
            
            wandb.watch(model, log_freq=100)
            best_epoch = 0
            for e in range(config.epochs):             
                model , loss = train_epoch(e + config.epochs * i_contitnue, len(configurations) * config.epochs, model, config.phases, config.learning_rate, config.learning_rate_change, config.learning_rate_change_epoch, datasets, dev, lossfunction, writer, step_pos, TB_path, batch_count)
                if loss < best_loss:
                    print("Updating best model")
                    best_loss = loss
                    best_epoch = e
                    current_model = copy.deepcopy(model.state_dict())
                    
            torch.save(current_model, str(TB_path) + f"/round{i_contitnue}.pth")
            i_beta= i
            wandb.log({f"beta round{i_beta}": beta})
            if i > 10:
                change_rate = schedular_rate(i_beta)
                agent_round = 0 
                beta -= change_rate

            if beta <= 0:
                agent_round += 1
                beta = 0.0
            if agent_round == 2:
                break
            # wandb.log({f"beta round{i}": beta})
            wandb.finish()

    except Exception as e:
        raise e
    finally:
        # save model
        writer.flush()
        writer.close()
