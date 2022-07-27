#!/usr/bin/env python
from email.policy import default
from select import epoll

from datagen.RaceTrackLoader import RaceTracksDataset

from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from ResNet8 import ResNet8
import torchvision.models.densenet
import argparse
from tqdm import tqdm


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

device = 'cuda'
epochs = 100
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 10
batch_size = 32

TB_suffix = "run6"
loss_type = "MSE"
phases = ['train', 'val']
skipLastXImages = 600


parser = argparse.ArgumentParser("Arguments for training")
parser.add_argument('--project-basepath', '-pb', type=str, default="/home/micha/dev/ml/orb_imitation")
parser.add_argument('--dataset-basepath', '-db', type=str, default="/media/micha/eSSD/datasets")
parser.add_argument('--dataset-basename', '-n', type=str, default="X4Gates_Circles")

args = parser.parse_args()

project_basepath = args.project_basepath
dataset_basepath = args.dataset_basepath
dataset_basename = args.dataset_basename
# dataset_basename = "X4Gates_Circle_2"

# create path for run
TB_path = Path(project_basepath, f"runs/ResNet8_bs={batch_size}_lt={loss_type}_lr={learning_rate}_c={TB_suffix}")
if TB_path.exists():
    print("TB_path exists")
    exit(0)
writer = SummaryWriter(str(TB_path))

print("loading dataset...")

train_set = RaceTracksDataset(
                dataset_basepath,
                dataset_basename,
                device=device,
                maxTracksLoaded=6,
                imageScale=100,
                skipTracks=0,
                grayScale=False,
                skipLastXImages=skipLastXImages
            )

print(len(train_set))

val_set = RaceTracksDataset(
                dataset_basepath,
                dataset_basename,
                device=device,
                maxTracksLoaded=3,
                imageScale=100,
                skipTracks=0,
                grayScale=False,
                skipLastXImages=skipLastXImages
            )

datasets = {
    'train':
        torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        ),
    'val':
        torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False
        )
}

dev = torch.device(device)

# model = ImageCNN4(device)
# model = dn.DenseNetCustom()
model = ResNet8(input_dim=3, output_dim=4, f=.5)
model = model.to(dev)


# if device == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

def schedule(epoch):
    """ Schedule learning rate according to epoch # """
    return learning_rate * learning_rate_change ** int(epoch / learning_rate_change_epoch)


# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


lossfunction = nn.MSELoss()

batch_count = {}
step_pos = {}
for el in phases:
    batch_count[el] = len(datasets[el])
    step_pos[el] = 0
    print(f"batch count {el}: {batch_count[el]}")

# print("batch count:", len(train_loader))

summary(model, (3, 200, 300), device=device)

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

            for images, labels in tqdm(dataset):  # change images to batches

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
                best_model = copy.deepcopy(model.state_dict())
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
