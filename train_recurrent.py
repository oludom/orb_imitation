#!/usr/bin/env python
from email.policy import default
from select import epoll

from matplotlib import image

from datagen.RaceTrackLoader import RaceTracksDataset, RecurrentRaceTrackDataset,RecurrentRaceTracksLoader

from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from models.racenet8 import RaceNet8
from models.ResNet8 import ResNet8
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


loss_type = "MSE"
phases = ['train', 'val']
skipLastXImages = 0
skipFirstXImages = 0

parser = argparse.ArgumentParser("Arguments for training")
parser.add_argument('--project-basepath', '-pb', type=str, default="/home/micha/dev/ml/orb_imitation")
parser.add_argument('--dataset-basepath', '-db', type=str, default="/media/micha/eSSD/datasets")
parser.add_argument('--dataset-basename', '-n', type=str, default="X4Gates_Circles")
parser.add_argument('--jobs', '-j', type=int, default=4)
parser.add_argument('--run', '-r', type=str, default='run0')
parser.add_argument('--frame', '-f', type=str, choices=['body', 'world'],default='world')

args = parser.parse_args()

project_basepath = args.project_basepath
dataset_basepath = args.dataset_basepath
dataset_basename = args.dataset_basename

# create path for run
TB_suffix = args.run
TB_path = Path(project_basepath, f"runs/RaceNet8_onegatel1108_Worldframe={batch_size}_lt={loss_type}_lr={learning_rate}_c={TB_suffix}")
if TB_path.exists():
    print("TB_path exists")
    exit(0)
writer = SummaryWriter(str(TB_path))

if args.frame =='world':
    from datagen.WorldTrackLoader import RaceTracksDataset, RecurrentRaceTrackDataset
elif args.frame=='body':
    from datagen.RaceTrackLoader import RaceTracksDataset,RecurrentRaceTrackDataset
print("loading dataset...")
train_tracks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
# train_tracks = [7,0,4,9,6,1,5, 8, 3]
# val_tracks = [8,3]
# train_tracks = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18]
# val_tracks = [19,20,21,22,23]
dataset = RecurrentRaceTrackDataset(
                dataset_basepath,
                dataset_basename,
                device=device,
                maxTracksLoaded=len(train_tracks),
                imageScale=100,
                skipTracks=0,
                grayScale=False,
                skipFirstXImages = skipFirstXImages,
                skipLastXImages=skipLastXImages,
                tracknames=train_tracks
            )

print(len(dataset))
split_ratio = 0.8
train_size = int(split_ratio * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

# train_set = RecurrentRaceTrackDataset(
#                 dataset_basepath,
#                 dataset_basename,
#                 device=device,
#                 maxTracksLoaded=len(train_tracks),
#                 imageScale=100,
#                 skipTracks=0,
#                 grayScale=False,
#                 skipLastXImages=skipLastXImages,
#                 tracknames=train_tracks
#             )

# print(len(train_set))

# val_set = RecurrentRaceTrackDataset(
#                 dataset_basepath,
#                 dataset_basename,
#                 device=device,
#                 maxTracksLoaded=len(val_tracks),
#                 imageScale=100,
#                 skipTracks=len(train_tracks),
#                 grayScale=False,
#                 skipLastXImages=skipLastXImages,
#                 train=False,
#                 tracknames=val_tracks
#             )

datasets = {
    'train':
        torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers = args.jobs
        ),
    'val':
        torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers = args.jobs
        )
}

dev = torch.device(device)

# model = ImageCNN4(device)
# model = dn.DenseNetCustom()
model = RaceNet8(input_dim=3, output_dim=4, f=.5)
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

# summary(model, (3, 200, 300), device=device)

best_model = copy.deepcopy(model.state_dict())
best_loss = 0.0

try:
    for epoch in range(epochs):
        total_loss = {}
        for el in phases:
            total_loss[el] = 0

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # print('TB')
        optimizer = optim.Adam(model.parameters(), lr=schedule(epoch), weight_decay=2e-4)

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            dataset = datasets[phase]

            for images, labels in tqdm(dataset):  # change images to batches
                # print(len(dataset))
                if epoch == 0 and step_pos['train'] == 0:
                    # print(len(images))
                    img_grid = torchvision.utils.make_grid(images[:, 0, :, :, :])
                    # img_grid = img_grid.permute(1,2,0)
                    writer.add_image('first images', img_grid)

                images = images.to(dev)
                labels = labels.to(dev)
                preds = torch.zeros_like(labels).to(dev)
                h_last = torch.zeros(1,len(images),256).to(dev)
                with torch.set_grad_enabled(phase == 'train'):
                    for i in range(images.size(1)): #sequence
                        # track history only if in train phase
                            # predict and calculate loss
                            preds[:, i, :] , h_now = model(images[:, i, :, :, :], h_last)
                            h_now = h_last                        
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
            print(phase)
            print(batch_count[phase])
            print(total_loss[phase])
            avg_total_loss = total_loss[phase] / batch_count[phase]
            print("epoch:", epoch, phase, "loss:", avg_total_loss)
            writer.add_scalar("Loss/epoch/" + phase, avg_total_loss, global_step=epoch)

            if phase == 'val' and avg_total_loss < best_loss:
                best_loss = avg_total_loss
                best_model = copy.deepcopy(model.state_dict())
    # writer.add_hparams({'lr': learning_rate, 'batch': batch_size},
    #                     {'loss': avg_total_loss}
    #                    )
    # print("---------------------------")
except Exception as e:
    raise e
finally:
    # save model
    writer.flush()
    writer.close()

    torch.save(best_model, str(TB_path) + "/best.pth")
