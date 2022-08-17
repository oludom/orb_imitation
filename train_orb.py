#!/usr/bin/env python
from select import epoll

from torch.backends import cudnn

from datagen.RaceTrackLoader import RaceTracksDataset

from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision.transforms as transforms

from ResNet8 import ResNet8
import torchvision.models.densenet

from calculate_mean_std import calculate_mean_std

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
# torch.multiprocessing.set_start_method('spawn')

device = 'cpu'
epochs = 100
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 5
batch_size = 32
resnet_factor = 0.25
num_train_tracks = 37
num_val_tracks = 15
jobs = 8

input_channels = {
    'rgb': False,
    'depth': True,
    'orb': True,
}

TB_suffix = "run0"
loss_type = "MSE"
phases = ['train', 'val']
skipFirstXImages = 0  # 60
skipLastXImages = 25  # 54

# project_basepath = "/workspaces/imitation"
project_basepath = "/home/micha/dev/ml/orb_imitation"
# dataset_basepath = "/media/micha/eSSD/datasets"
# dataset_basepath = "/home/micha/dev/datasets/droneracing"
dataset_basepath = "/data/datasets"
# dataset_basename = "X4Gates_Circle_right_"
# dataset_basename = "X4Gates_Circles"
dataset_basename = "X1GateDepth"
# dataset_basename = "X4Gates_Circle_2"

num_input_channels = (input_channels['rgb'] * 3) + \
                     (input_channels['depth'] * 1) + \
                     (input_channels['orb'] * 1)

if num_input_channels < 1:
    print("No input channels selected")
    exit(0)



# create path for run
itypes = [
    'rgb' if input_channels['rgb'] else '',
    'd' if input_channels['depth'] else '',
    'o' if input_channels['orb'] else ''
]
itypes = ''.join(itypes)
TB_path = Path(project_basepath, f"runs/ResNet8_l={itypes}_f={resnet_factor}_bs={batch_size}_lt={loss_type}_lr={learning_rate}_c={TB_suffix}")
if TB_path.exists():
    print("TB_path exists")
    exit(0)
writer = SummaryWriter(str(TB_path))

tf = None
if itypes == 'rgb' or itypes == 'rgbo':
    tf = transforms.Compose([
        transforms.Normalize(
            (0.4694, 0.4853, 0.4915),
            (0.2693, 0.2981, 0.3379))
    ])
elif itypes == 'rgbd' or itypes == 'rgbdo':
    tf = transforms.Compose([
        transforms.Normalize(
            (0.4694,  0.4853,  0.4915, 49.0911),
            (2.6934e-01, 2.9809e-01, 3.3785e-01, 6.9964e+02))
    ])
elif itypes == 'd' or itypes == 'do':
    tf = transforms.Compose([
        transforms.Normalize(
            (49.0911),
            (6.9964e+02))
    ])



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

dev = torch.device(device)

model = ResNet8(input_dim=num_input_channels, output_dim=4, f=resnet_factor)
model = model.to(dev)


# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

def schedule(epoch):
    """ Schedule learning rate according to epoch # """
    return learning_rate * learning_rate_change ** int(epoch / learning_rate_change_epoch)


lossfunction = nn.MSELoss()

batch_count = {}
step_pos = {}
for el in phases:
    batch_count[el] = len(datasets[el])
    step_pos[el] = 0
    print(f"batch count {el}: {batch_count[el]}")

# print("batch count:", len(train_loader))

summary(model, (num_input_channels, 144, 256), device=device)

# print mean and std of dataset
# mean, std = calculate_mean_std(datasets['train'])
# print("mean:", mean, "std:", std)
# exit(0)

best_model = copy.deepcopy(model.state_dict())
best_loss = 0.0

try:
    for epoch in range(epochs):
        total_loss = {}
        for el in phases:
            total_loss[el] = 0

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        optimizer = optim.Adam(model.parameters(), lr=schedule(epoch), weight_decay=2e-4)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            dataset = datasets[phase]

            for images, labels in dataset:  # change images to batches

                if epoch == 0 and step_pos['train'] == 0:
                    img_grid = torchvision.utils.make_grid(images)
                    # img_grid = img_grid.permute(1,2,0)
                    writer.add_image('first images', img_grid)

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

            if phase == 'val' and avg_total_loss < best_loss:
                best_loss = avg_total_loss
            current_model = copy.deepcopy(model.state_dict())
            torch.save(current_model, str(TB_path) + f"/epoch{epoch}.pth")

    # writer.add_hparams({'lr': learning_rate, 'batch': batch_size},
    #                    # {'loss': avg_total_loss}
    #                    )
    # print("---------------------------")
except Exception as e:
    raise e
finally:
    # save model
    writer.flush()
    writer.close()

    torch.save(best_model, str(TB_path) + "/best.pth")
