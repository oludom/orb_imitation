import sys

from sklearn.utils import shuffle

sys.path.insert(0, './datagen')

from DaggerClient import DaggerClient
from train import *  # also imports config

from copy import deepcopy
import torch

from ResNet8 import ResNet8
import torch.backends.cudnn as cudnn
import torch.nn as nn

if __name__ == "__main__":

    import contextlib

    beta = 1

    # tb path handling
    TB_path, writer = get_path_for_run(config.project_basepath, config.dataset_basename, config.itypes,
                                       config.resnet_factor, config.batch_size, config.loss_type, config.learning_rate,
                                       config.TB_suffix)

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
    best_loss = 0.0

    try:
        # for each configuration
        for i, gateConfig in enumerate(configurations):

            # use i + 1 if model is not pretrained
            beta = 1 / (i+2)

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
                                
            model = train_epoch(i + config.epoch_start, len(configurations) + config.epoch_start, model, config.phases, config.learning_rate, config.learning_rate_change, config.learning_rate_change_epoch, datasets, dev, lossfunction, writer, step_pos, TB_path, batch_count)
            

    except Exception as e:
        raise e
    finally:
        # save model
        writer.flush()
        writer.close()
