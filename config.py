

import torchvision.transforms as transforms

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
    'rgb': True,
    'depth': False,
    'orb': False,
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
