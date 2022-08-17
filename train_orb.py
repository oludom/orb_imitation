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
from config import *

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
# torch.multiprocessing.set_start_method('spawn')



TB_path = Path(project_basepath, f"runs/ResNet8_l={itypes}_f={resnet_factor}_bs={batch_size}_lt={loss_type}_lr={learning_rate}_c={TB_suffix}")
if TB_path.exists():
    print("TB_path exists")
    exit(0)
writer = SummaryWriter(str(TB_path))



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