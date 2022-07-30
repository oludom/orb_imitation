import torch
import torch.nn as nn
import torch.optim as optim
from models.racenet8 import RaceNet8
from models.ResNet8 import ResNet8

from datagen.RaceTrackLoader import RecurrentRaceTrackDataset
from pathlib import Path
import argparse


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

device = 'cuda'
epochs = 100
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 10
batch_size = 32

TB_suffix = "run0"
loss_type = "MSE"
phases = ['train', 'val']
skipLastXImages = 0

parser = argparse.ArgumentParser("Arguments for training")
parser.add_argument('--project-basepath', '-pb', type=str, default="/home/micha/dev/ml/orb_imitation")
parser.add_argument('--dataset-basepath', '-db', type=str, default="/media/micha/eSSD/datasets")
parser.add_argument('--dataset-basename', '-n', type=str, default="X4Gates_Circles")
parser.add_argument('--jobs', '-j', type=int, default=4)
parser.add_argument('--run', '-r', type=str, default='run0')

args = parser.parse_args()

project_basepath = args.project_basepath
dataset_basepath = args.dataset_basepath
dataset_basename = args.dataset_basename


print("loading dataset...")

train_tracks = [0,1,2,3,4,5,6,7]
val_tracks = [8,9]

datasets = {
    'train':
        torch.utils.data.DataLoader(
            RecurrentRaceTrackDataset(
                dataset_basepath,
                dataset_basename,
                device=device,
                maxTracksLoaded=12,
                imageScale=100,
                skipTracks=0,
                grayScale=False,
                skipLastXImages=skipLastXImages
            ),
            batch_size=batch_size,
            shuffle=True
        ),
    'val':
        torch.utils.data.DataLoader(
            RecurrentRaceTrackDataset(
                dataset_basepath,
                dataset_basename,
                device=device,
                maxTracksLoaded=6,
                imageScale=100,
                skipTracks=12,
                grayScale=False,
                skipLastXImages=skipLastXImages
            ),
            batch_size=batch_size,
            shuffle=True
        )
}

dev = torch.device(device)

for phase in phases:

    dataset = datasets[phase]

    for images, labels in dataset:  # change images to batches

        images = images.to(dev)
        labels = labels.to(dev)
        print(images.size())
        print(labels.size())