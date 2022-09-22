import torchvision.transforms as transforms

device = 'cuda'
parallel = True
epochs = 5
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 5
batch_size = 32
resnet_factor = 0.5
num_train_tracks = 170
num_val_tracks = 50
jobs = 8


input_channels = {
    'rgb': True,
    'depth': True,
    'orb': True
}

TB_suffix = "run0"
loss_type = "MSE"
phases = ['train', 'val']
skipFirstXImages = 80  # 60
skipLastXImages = 10  # 54

# project_basepath = "/workspaces/imitation"
project_basepath = "/home/micha/dev/ml/orb_imitation"
# dataset_basepath = "/media/micha/eSSD/datasets"
# dataset_basepath = "/home/micha/dev/datasets/droneracing"
dataset_basepath = "/data/datasets/dr_pretrain"
# dataset_basename = "X4Gates_Circle_right_"
# dataset_basename = "X4Gates_Circles"
dataset_basename = "dr_pretrain"
# dataset_basename = "X4Gates_Circle_2"

# X1Gate200
# dataset_mean = (0.4697,  0.4897,  0.4988, 49.4976)
# dataset_std = (2.7118e-01, 2.9868e-01, 3.3708e-01, 6.8829e+02)

# X4Gates_Circle_right
# dataset_mean = (0.4699,  0.4793,  0.4848, 67.2920)
# dataset_std = (2.5673e-01, 2.9010e-01, 3.2995e-01, 7.7903e+02)

# X1Gate8tracks
# dataset_mean = (0.4660,  0.4733,  0.4792, 78.8772)
# dataset_std = (2.5115e-01, 2.8758e-01, 3.2971e-01, 8.9808e+02)

# domain randomization pretrain
dataset_mean = (0.4973,  0.4651,  0.4801, 32.6839)
dataset_std = (0.1714,  0.1960,  0.2171, 22.7819)

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


