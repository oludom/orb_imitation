

import torchvision.transforms as transforms

device = 'cuda'
parallel = True
epochs = 10
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 5
batch_size = 32
resnet_factor = 0.25
num_train_tracks = 20
num_val_tracks = 5
jobs = 1

input_channels = {
    'rgb': True,
    'depth': False,
    'orb': False,
    'gray': False
}

TB_suffix = "run1"
loss_type = "MSE"
phases = ['train', 'val']
skipFirstXImages = 0  # 60
skipLastXImages = 0  # 54

# project_basepath = "/workspaces/imitation"
project_basepath = "/home/micha/dev/ml/orb_imitation"
# dataset_basepath = "/media/micha/eSSD/datasets"
# dataset_basepath = "/home/micha/dev/datasets/droneracing"
dataset_basepath = "/data/datasets"
# dataset_basename = "X4Gates_Circle_right_"
# dataset_basename = "X4Gates_Circles"
dataset_basename = "X4Gates_Circle_right"
# dataset_basename = "X4Gates_Circle_2"

# X1Gate200
# dataset_mean = (0.4697,  0.4897,  0.4988, 49.4976)
# dataset_std = (2.7118e-01, 2.9868e-01, 3.3708e-01, 6.8829e+02)

# X4Gates_Circle_right
dataset_mean = (0.4699,  0.4793,  0.4848, 67.2920)
dataset_std = (2.5673e-01, 2.9010e-01, 3.2995e-01, 7.7903e+02)

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
            dataset_mean[:3],
            dataset_std[:3])
    ])
elif itypes == 'rgbd' or itypes == 'rgbdo':
    tf = transforms.Compose([
        transforms.Normalize(
            dataset_mean,
            dataset_std)
    ])
elif itypes == 'd' or itypes == 'do':
    tf = transforms.Compose([
        transforms.Normalize(
            dataset_mean[3],
            dataset_std[3])
    ])
