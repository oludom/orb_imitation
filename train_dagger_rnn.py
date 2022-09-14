import math
import sys

from sklearn.utils import shuffle

sys.path.insert(0, './datagen')

# from DaggerClient import DaggerClient
from datagen.DaggerClientRNN import DaggerClient
from train_rnn import *  # also imports config

from copy import deepcopy
import torch
from RaceNet8 import RaceNet8
from ResNet8 import ResNet8
import torch.backends.cudnn as cudnn
import torch.nn as nn

if __name__ == "__main__":

    import contextlib
    # beta scheduler
    beta = 1
    weight = 0.1
    beta_rate = 0.38
    def schedular_rate(round):
        return (weight) /((round +1) * beta_rate)

    # tb path handling
    TB_path, writer = get_path_for_run_rnn(config.project_basepath, config.dataset_basename, config.itypes,
                                       config.resnet_factor, config.batch_size, config.loss_type, config.learning_rate,
                                       config.TB_suffix,config.beta_rate)

    dev = torch.device(config.device)

    # model = ResNet8(input_dim=config.num_input_channels, output_dim=4, f=config.resnet_factor)
    # model = model.to(dev)
    ### Recurrent model

    model = RaceNet8(input_dim=config.num_input_channels, output_dim=4, f=config.resnet_factor)
    print(config.num_input_channels)
    model = model.to(dev)

    # if config.parallel and config.device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True

    if config.parallel and config.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        
    lossfunction = nn.MSELoss()

    # load initial model 
    if config.initial_weight_path:
        model.load_state_dict(torch.load(config.initial_weight_path))

    with contextlib.closing(DaggerClient(model, configFilePath='datagen/config_cleft.json', createDataset=False)) as dc:
        # generate random gate configurations within bounds set in config.json
        dc.generateGateConfigurations()
        configurations1 = deepcopy(dc.gateConfigurations)

    with contextlib.closing(
            DaggerClient(model, configFilePath='datagen/config_cright.json', createDataset=False)) as dc:
        # generate random gate configurations within bounds set in config.json
        dc.generateGateConfigurations()
        configurations2 = deepcopy(dc.gateConfigurations)

    configurations = list(configurations1) + list(configurations2)

    configurations = shuffle(configurations)

    step_pos = {}
    for el in config.phases:
        step_pos[el] = 0

    best_model = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    agent_round = 0
    try:
        # for each configuration
        for i, gateConfig in enumerate(configurations1):
            wandb.init(project="dagger_rgbdo", entity="dungtd2403")
            wandb.config = {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 32
            }


            with contextlib.closing(
                    DaggerClient(model, beta=beta, device=config.device, raceTrackName=f"track{i + config.skip_tracks}",
                                 configFilePath='datagen/config.json', createDataset=True)) as dc:
                dc.gateConfigurations = [gateConfig]

                # load next gate arrangement 
                dc.loadNextGatePosition()

                # fly mission
                dc.run(showMarkers=False, uav_position=dc.config.uav_position)

            datasets = load_dataset_train_val_split_recurrent(config.dataset_basepath, config.dataset_basename, config.device,
                                                    1000,
                                                    config.input_channels, config.skipFirstXImages,
                                                    config.skipLastXImages, config.batch_size, config.tf, config.jobs)

            batch_count = {}
            for el in config.phases:
                batch_count[el] = len(datasets[el])
                print(f"batch count {el}: {batch_count[el]}")


            wandb.watch(model, log_freq=100)
            for e in range(1):
                model , loss = train_epoch_recurrent(e, i, model, config.phases, config.learning_rate,
                                    config.learning_rate_change, config.learning_rate_change_epoch, datasets, dev,
                                    lossfunction, writer, step_pos, TB_path, batch_count)
                # if loss < best_loss:
                #     print("Updating best model")
                #     best_loss = loss
                #     best_epoch = e
                current_model = copy.deepcopy(model.state_dict())
                    
            torch.save(current_model, str(TB_path) + f"/round{i}.pth")
            
            if beta > 0:
                change_rate = schedular_rate(i)
                beta -= change_rate
            if beta <= 0:
                agent_round += 1
                beta = 0.0001
            if agent_round == 3:
                break
            wandb.log({f"beta round{i}": beta})
            wandb.finish()

    except Exception as e:
        raise e
    finally:
        # save model
        writer.flush()
        writer.close()
