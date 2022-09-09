#!/usr/bin/env python

from torch.backends import cudnn

from datagen.RaceTrackLoader import RaceTracksDataset

from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import torchvision.transforms as transforms

from ResNet8 import ResNet8
import torchvision.models.densenet

from calculate_mean_std import calculate_mean_std
import config

import argparse

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


# torch.multiprocessing.set_start_method('spawn')

def load_dataset_train_val_split(dataset_basepath, dataset_basename, device, maxTracks, input_channels,
                 skipFirstXImages, skipLastXImages, batch_size, tf, jobs=1):

    dataset = RaceTracksDataset(
                    dataset_basepath,
                    dataset_basename,
                    device=device,
                    maxTracksLoaded=maxTracks,
                    imageScale=100,
                    skipTracks=0,
                    grayScale=False,
                    imageTransforms=tf,
                    skipLastXImages=skipLastXImages,
                    skipFirstXImages=skipFirstXImages,
                    loadRGB=input_channels['rgb'],
                    loadDepth=input_channels['depth'],
                    loadOrb=input_channels['orb']
                )
    phases = ['train', 'val']
    # print(len(dataset))
    split_ratio = 0.8
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])


    datasets = {
        'train':
            torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                # num_workers = jobs
            ),
        'val':
            torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=True,
                # num_workers = jobs
            )
    }
    return datasets

def load_dataset(dataset_basepath, dataset_basename, device, num_train_tracks, num_val_tracks, input_channels,
                 skipFirstXImages, skipLastXImages, batch_size, tf, jobs=1):
    print("loading dataset...")

    datasets = {
        'train':
            torch.utils.data.DataLoader(
                RaceTracksDataset(
                    dataset_basepath,
                    dataset_basename,
                    device=device,
                    maxTracksLoaded=num_train_tracks,
                    imageScale=100,
                    skipTracks=0,
                    grayScale=False,
                    imageTransforms=tf,
                    skipLastXImages=skipLastXImages,
                    skipFirstXImages=skipFirstXImages,
                    loadRGB=input_channels['rgb'],
                    loadDepth=input_channels['depth'],
                    loadOrb=input_channels['orb']
                ),
                batch_size=batch_size,
                shuffle=True,
                # num_workers=jobs
            ),
        'val':
            torch.utils.data.DataLoader(
                RaceTracksDataset(
                    dataset_basepath,
                    dataset_basename,
                    device=device,
                    maxTracksLoaded=num_val_tracks,
                    imageScale=100,
                    skipTracks=num_train_tracks,
                    grayScale=False,
                    imageTransforms=tf,
                    skipLastXImages=skipLastXImages,
                    skipFirstXImages=skipFirstXImages,
                    loadRGB=input_channels['rgb'],
                    loadDepth=input_channels['depth'],
                    loadOrb=input_channels['orb']
                ),
                batch_size=batch_size,
                shuffle=True,
                # num_workers=jobs
            )
    }

    return datasets


def schedule(epoch, learning_rate, learning_rate_change, learning_rate_change_epoch):
    """ Schedule learning rate according to epoch # """
    return learning_rate * learning_rate_change ** int(epoch / learning_rate_change_epoch)


def print_mean_std(datasets):
    # print mean and std of dataset
    mean, std = calculate_mean_std(datasets['train'])
    print("mean:", mean, "std:", std)
    # exit(0)

def get_path_for_run(project_basepath, dataset_basename, itypes, resnet_factor, batch_size, loss_type, learning_rate, TB_suffix):
    TB_path = Path(project_basepath,
                f"runs/ResNet8_ds={dataset_basename}_l={itypes}_f={resnet_factor}"
                f"_bs={batch_size}_lt={loss_type}_lr={learning_rate}_c={TB_suffix}")
    if TB_path.exists():
        print("TB_path exists")
        exit(0)
    writer = SummaryWriter(str(TB_path))
    return TB_path, writer


def train_epoch(epoch, epochs, model, phases, learning_rate, learning_rate_change, learning_rate_change_epoch, datasets, dev, lossfunction, writer, step_pos, TB_path, batch_count):

    total_loss = {}
    for el in phases:
        total_loss[el] = 0

    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    optimizer = optim.Adam(model.parameters(),
                            lr=schedule(epoch, learning_rate, learning_rate_change, learning_rate_change_epoch),
                            weight_decay=2e-4
                            )

    for phase in phases:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        dataset = datasets[phase]

        for images, labels in dataset:  # change images to batches

            # if epoch == 0 and step_pos['train'] == 0:
            #     img_grid = torchvision.utils.make_grid(images)
            #     # img_grid = img_grid.permute(1,2,0)
            #     writer.add_image('first images', img_grid)

            images = images.to(dev)
            labels = labels.to(dev)

            # track history only if in train phase
            with torch.set_grad_enabled(phase == 'train'):
                # predict and calculate loss
                preds = model(images)

                loss = lossfunction(preds, labels)

                # only backward and optimize if in training phase
                if phase == 'train':
                    # calculate gradients
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()

            # print("batch", image_count, "loss", loss.item())
            total_loss[phase] += loss.item()

            # print("batch:", i, "loss:", loss.item())
            writer.add_scalar(f"Loss/{phase}", loss.item(), global_step=(step_pos[phase]))
            step_pos[phase] += 1

        # step_lr_scheduler.step()
        avg_total_loss = total_loss[phase] / batch_count[phase]
        print("epoch:", epoch, phase, "loss:", avg_total_loss)
        writer.add_scalar("Loss/epoch/" + phase, avg_total_loss, global_step=epoch)

        # if phase == 'val' and avg_total_loss < best_loss:
        #     best_loss = avg_total_loss
        current_model = copy.deepcopy(model.state_dict())
        torch.save(current_model, str(TB_path) + f"/epoch{epoch}.pth")

    return model



def train(project_basepath,
          dataset_basepath,
          dataset_basename,
          device,
          parallel,
          epochs,
          learning_rate,
          learning_rate_change,
          learning_rate_change_epoch,
          batch_size,
          resnet_factor,
          num_train_tracks,
          num_val_tracks,
          jobs,
          input_channels,
          TB_suffix,
          loss_type,
          phases,
          skipFirstXImages,
          skipLastXImages,
          num_input_channels,
          itypes,
          tf,  # image transforms
          *args,
          **kwargs
          ):

    # tb path handling
    TB_path, writer = get_path_for_run(project_basepath, dataset_basename, itypes, resnet_factor, batch_size, loss_type, learning_rate, TB_suffix)

    datasets = load_dataset(dataset_basepath, dataset_basename, device, num_train_tracks, num_val_tracks,
                            input_channels, skipFirstXImages, skipLastXImages, batch_size, tf, jobs)

    dev = torch.device(device)

    model = ResNet8(input_dim=num_input_channels, output_dim=4, f=resnet_factor)
    model = model.to(dev)

    if parallel and device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    lossfunction = nn.MSELoss()

    # print stats
    batch_count = {}
    step_pos = {}
    for el in phases:
        batch_count[el] = len(datasets[el])
        step_pos[el] = 0
        print(f"batch count {el}: {batch_count[el]}")
    # summary(model, (num_input_channels, 144, 256), device=device)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    try:
        for epoch in range(epochs):
            # train epoch
            current_loss, model = train_epoch(epoch, epochs, model, phases, learning_rate, learning_rate_change, learning_rate_change_epoch, datasets, dev, lossfunction, writer, step_pos, TB_path, batch_count)
                

    except Exception as e:
        raise e
    finally:
        # save model
        writer.flush()
        writer.close()

        # torch.save(best_model, str(TB_path) + "/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_config", type=bool, default=True)
    parser.add_argument("--project_basepath", type=str, default="/home/alexander/PycharmProjects/DeepLearningProject")
    parser.add_argument("--dataset_basepath", type=str,
                        default="/home/alexander/PycharmProjects/DeepLearningProject/datasets")
    parser.add_argument("--dataset_basename", type=str, default="dataset_2020-03-11_14-13-17")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_rate_change", type=float, default=0.1)
    parser.add_argument("--learning_rate_change_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resnet_factor", type=int, default=1)
    parser.add_argument("--num_train_tracks", type=int, default=100)
    parser.add_argument("--num_val_tracks", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--input_channels", type=str, default="rgb,depth,orb")
    parser.add_argument("--TB_suffix", type=str, default="test")
    parser.add_argument("--loss_type", type=str, default="MSE")
    parser.add_argument("--phases", type=str, default="train,val")
    parser.add_argument("--skipFirstXImages", type=int, default=0)
    parser.add_argument("--skipLastXImages", type=int, default=0)
    parser.add_argument("--num_input_channels", type=int, default=3)
    parser.add_argument("--itypes", type=str, default="rgb,depth,orb")

    args = parser.parse_args()
    if args.use_config:
        train(**vars(config))
    else:
        train(**vars(args))
