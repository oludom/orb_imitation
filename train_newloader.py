#!/usr/bin/env python
from email.policy import default
from select import epoll
from termios import N_MOUSE
# import wandb
import torchvision.transforms as transforms
from pathlib import Path
import copy
from tokenize import Double

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
# from torchsummary import summary
from models.racenet8 import RaceNet8
from models.ResNet8 import ResNet8
import torchvision.models.densenet
import argparse
from tqdm import tqdm


def train_main(args, run : int):
    learning_rate = 0.001
    learning_rate_change = 0.1
    learning_rate_change_epoch = 10
    batch_size = 32
    run = run
    loss_type = "MSE"
    torch.set_printoptions(linewidth=120)
    torch.set_grad_enabled(True)

    # create wandb project

#     wandb.init(project="DAgger", entity="dungtd2403")
#     wandb.config = {
#     "learning_rate": 0.001,
#     "epochs": 50,
#     "batch_size": 32
# }
    device = 'cuda'
    epochs = 50
    
    project_basepath = args.project_basepath
    dataset_basepath = args.dataset_basepath
    dataset_basename = args.dataset_basename
    skipFirstXImages = 3
    skipLastXImages = 5 
    # create path for run
    # TB_suffix = args.run
    TB_path = Path(project_basepath, f"run_normalized_dagger/DAgger_ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_run{run}")
    if TB_path.exists():
        print("TB_path exists")
        exit(0)
    writer = SummaryWriter(str(TB_path))


    if args.frame =='world':
        from datagen.MyTrackLoader import RaceTracksDataset
    elif args.frame=='body':
        from datagen.RaceTrackLoader import RaceTracksDataset

    print("loading dataset...")

    train_tracks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    # val_tracks = [8,3]
    maxTrackLoaded = 180


    dataset = RaceTracksDataset(
                    dataset_basepath,
                    dataset_basename,
                    device=device,
                    maxTracksLoaded=maxTrackLoaded,
                    imageScale=100,
                    skipTracks=0,
                    grayScale=False,
                    skipFirstXImages = skipFirstXImages,
                    skipLastXImages= skipLastXImages,
                    tracknames=train_tracks
                )
    phases = ['train', 'val']
    print(len(dataset))
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

    model = ResNet8(input_dim=3, output_dim=4, f=1)
    model = model.to(dev)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained))
    # wandb.watch(model, log_freq=100)

    def schedule(epoch):
        """ Schedule learning rate according to epoch # """
        return learning_rate * learning_rate_change ** int(epoch / learning_rate_change_epoch)

    def mean_std(loader):
        images, labels = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean, std = images.mean([0,2,3]), images.std([0,2,3])
        return mean, std

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
                mean , std = mean_std(dataset)
                print(f"mean : {mean}, std: {std}")

                num_batch = 0

                for step, (images, labels )in enumerate(tqdm(dataset)):  # change images to batches

                    if epoch == 0 and step_pos['train'] == 0:
                        img_grid = torchvision.utils.make_grid(images)
                        # img_grid = img_grid.permute(1,2,0)
                        writer.add_image('first images', img_grid)
                    images = transforms.Compose([
                        transforms.Normalize(
                        mean,
                        std)
                     ])(images)
                    images = images.to(dev)
                    labels = labels.to(dev)

                    # track history only if in train phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # predict and calculate loss
                        preds = model(images)
                        # print(len(preds))
                        # print(f'pred_x = {preds[0]}, pred_y = {preds[1]}, pred_z = {preds[2]}, pred_yaw = {preds[3]}')
                        # print(f'label_x = {labels[0]}')
                        # loss = lossfunction(preds, labels)
                        loss = lossfunction(preds, labels)
                        # print(f'loss = {loss}')
                        # only backward and optimize if in training phase
                        if phase == 'train':
                            # calculate gradients
                            optimizer.zero_grad()
                            loss.backward()
                            # update weights
                            optimizer.step()
                            
                    # print("batch", image_count, "loss", loss.item())
                    total_loss[phase] += loss.item()
                    
                    # print("batch:", num_batch, "loss:", loss.item())
                    
                    num_batch +=1
                    writer.add_scalar(f"Loss/{phase}", loss.item(), global_step=(step_pos[phase]))
                    step_pos[phase] += 1
                    n_steps_per_epoch = len(dataset)
                    metrics = {f"{phase}/batch_loss": loss, 
                                f"{phase}/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                            }
                    
                    # if step + 1 < n_steps_per_epoch:
                    #     # 🐝 Log train metrics to wandb 
                    #     wandb.log(metrics)

                # step_lr_scheduler.step()
                avg_total_loss = total_loss[phase] / batch_count[phase]
                print("epoch:", epoch, phase, "loss:", avg_total_loss)
                writer.add_scalar("Loss/epoch/" + phase, avg_total_loss, global_step=epoch)
                
                # wandb.log({f"{phase}/avg_loss": avg_total_loss})
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
        # wandb.finish()

if __name__ =='__main__':
    torch.set_printoptions(linewidth=120)
    torch.set_grad_enabled(True)
    
    device = 'cuda'
    epochs = 50
    learning_rate = 0.001
    learning_rate_change = 0.1
    learning_rate_change_epoch = 10
    batch_size = 32


    loss_type = "MSE"
    phases = ['train', 'val']
    skipFirstXImages = 0
    skipLastXImages = 0


    parser = argparse.ArgumentParser("Arguments for training")
    parser.add_argument('--project-basepath', '-pb', type=str, default="/media/data2/teamICRA")
    parser.add_argument('--dataset-basepath', '-db', type=str, default="/media/data2/teamICRA/X4Gates_Circles_rl18tracks")
    parser.add_argument('--dataset-basename', '-n', type=str, default="X1Gate_dagger")
    parser.add_argument('--jobs', '-j', type=int, default=4)
    parser.add_argument('--run', '-r', type=str, default='run0')
    parser.add_argument('--frame', '-f', type=str, choices=['body', 'world'],default='world')

    args = parser.parse_args()
    run = 0
    train_main(args,run)