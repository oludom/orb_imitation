#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import message_filters as mf
from std_msgs.msg import String

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import math
import torch.backends.cudnn as cudnn
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

### config
import torchvision.transforms as transforms

device = 'cuda'
parallel = True
epochs = 5
learning_rate = 0.001
learning_rate_change = 0.1
learning_rate_change_epoch = 5
batch_size = 32
resnet_factor = 0.25
num_train_tracks = 170
num_val_tracks = 50
jobs = 8

input_channels = {
    'rgb': True,
    'depth': True,
    'orb': True
}

TB_suffix = "run10"
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
dataset_mean = (0.4973, 0.4651, 0.4801, 32.6839)
dataset_std = (0.1714, 0.1960, 0.2171, 22.7819)

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
        # transforms.Resize((144, 256)),
        transforms.Normalize(
            dataset_mean[:3],
            dataset_std[:3])
    ])
elif itypes == 'rgbd' or itypes == 'rgbdo':
    tf = transforms.Compose([
        # transforms.Resize((144, 256)),
        transforms.Normalize(
            dataset_mean,
            dataset_std)
    ])
elif itypes == 'd' or itypes == 'do':
    tf = transforms.Compose([
        # transforms.Resize((144, 256)),
        transforms.Normalize(
            dataset_mean[3],
            dataset_std[3])
    ])


# calculate magnitude of vector
def magnitude(vec):
    return np.sqrt(vec.dot(vec))


'''
taken from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
pytorch does not support padding='same' if Conv2d has stride other than 1
therefore use helperfunction to calculate padding
'''


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResNet8(nn.Module):
    def __init__(self, input_dim, output_dim, f=0.25):
        super(ResNet8, self).__init__()

        # kaiming he norm used as default by pytorch

        # first residual block
        self.x1 = nn.Sequential(
            Conv2dSame(in_channels=input_dim, out_channels=int(32 * f), kernel_size=(5, 5), stride=(2, 2)),
            nn.MaxPool2d((3, 3), stride=(2, 2))
        )
        self.x2 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(32 * f), out_channels=int(32 * f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(32 * f), out_channels=int(32 * f), kernel_size=(3, 3), stride=(1, 1))
        )
        self.x1_ = Conv2dSame(in_channels=int(32 * f), out_channels=int(32 * f), kernel_size=(1, 1), stride=(2, 2))

        # second residual block
        self.x4 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(32 * f), out_channels=int(64 * f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(64 * f), out_channels=int(64 * f), kernel_size=(3, 3), stride=(1, 1))
        )

        self.x3 = Conv2dSame(in_channels=int(32 * f), out_channels=int(64 * f), kernel_size=(1, 1), stride=(2, 2))

        # third residual block

        self.x6 = nn.Sequential(
            nn.ReLU(),
            Conv2dSame(in_channels=int(64 * f), out_channels=int(128 * f), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            Conv2dSame(in_channels=int(128 * f), out_channels=int(128 * f), kernel_size=(3, 3), stride=(1, 1))
        )

        self.x5 = Conv2dSame(in_channels=int(64 * f), out_channels=int(128 * f), kernel_size=(1, 1), stride=(2, 2))

        self.x7 = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.dense = nn.Linear(in_features=int(1280 * 4 * f), out_features=256)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim)
        )

    def forward(self, t):
        # first residual block
        t1 = self.x1(t)
        t2 = self.x2(t1)
        t1_ = self.x1_(t1)
        t3 = t2 + t1_

        # second residual block
        t4 = self.x4(t3)
        t3_ = self.x3(t3)
        t5 = t3_ + t4

        # third resudual block
        t6 = self.x6(t5)
        t5_ = self.x5(t5)
        t7 = t5_ + t6

        t8 = self.x7(t7)
        td = self.dense(t8)
        to = self.out(td)

        return to


def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_orb(img_left, img_right=None, n_features=1000, max_matches=100, orb=None):
    if len(img_left.shape) > 2:
        img_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    if orb is None:
        orb = cv.ORB_create(nfeatures=n_features)
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    if img_right is not None:
        kp_right, des_right = orb.detectAndCompute(img_right, None)
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_left, des_right)
        matches = sorted(matches, key=lambda x: x.distance)
        return kp_left, des_left, kp_right, des_right, matches[:max_matches]
    return kp_left, des_left, None, None, None


torch.set_grad_enabled(False)

now = lambda: int(round(time.time() * 1000))
pd = lambda s, t: print(f"{t}: {now() - s}ms")


class NetworkTestClient():

    def __init__(self, modelPath, raceTrackName="track0", device='cuda', *args, **kwargs):
        self.loaded = False

        # ros setup
        self.firstCall = True

        self.bridge = CvBridge()
        lisub = mf.Subscriber('/zedm/zed_node/left/image_rect_color', Image)
        risub = mf.Subscriber("/zedm/zed_node/depth/depth_registered", Image)

        self.ts = mf.TimeSynchronizer([lisub, risub], 10)
        self.ts.registerCallback(self.callback)

        self.debugPublisher = rospy.Publisher("/vel", String, queue_size=10)

        self.model = ResNet8(input_dim=num_input_channels, output_dim=4, f=resnet_factor)
        if device == 'cuda':
            self.model = nn.DataParallel(self.model)
            cudnn.benchmark = True

        self.model.load_state_dict(torch.load(modelPath))

        self.device = device
        self.dev = torch.device(device)

        self.model.to(self.dev)
        self.model.eval()

        print("model loaded.")
        self.loaded = True

    def preprocessImages(self, image, depth):

        sample = None
        withDepth = input_channels['depth']

        if input_channels['orb']:
            kp, des, _, _, _ = get_orb(image)

        # preprocess image
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        if input_channels['rgb']:
            sample = image

        if withDepth:
            depth = transforms.Compose([
                transforms.ToTensor(),
            ])(depth)
            if sample is not None:
                sample = torch.cat((sample, depth), dim=0)
            else:
                sample = depth

        if tf:
            sample = tf(sample)

        if input_channels['orb']:
            orbmask = torch.zeros_like(sample[0])
            for el in kp:
                x, y = el.pt
                orbmask[int(y), int(x)] = 1
            orbmask = orbmask.unsqueeze(0)
            if sample is not None:
                sample = torch.cat((sample, orbmask), 0)
            else:
                sample = orbmask

        return sample

    def callback(self, limsg, depthmsg) -> None:
        if not self.loaded:
            return
        if self.firstCall:
            rospy.loginfo("Camera connected, receiving images.")
            self.firstCall = False

        try:
            # convert image
            if input_channels['rgb'] or input_channels['orb']:
                lcvImage = self.bridge.imgmsg_to_cv2(limsg, "bgr8")
                lcvImage = image_resize(lcvImage, width=256, height=144)
            else:
                lcvImage = None

            if input_channels['depth']:
                depthImage = self.bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
                depthImage = image_resize(depthImage, width=256, height=144)
                depthImage = np.array(depthImage, dtype=np.float32)
                depthImage = np.nan_to_num(depthImage)
            else:
                depthImage = None

            sample = self.preprocessImages(lcvImage, depthImage)

            sample = torch.unsqueeze(sample, dim=0)
            sample = sample.to(torch.device("cuda"))

            # predict vector with network
            pred = self.model(sample)
            pred = pred.to(torch.device('cpu'))
            pred = pred.detach().numpy()
            pred = pred[0]  # remove batch

            self.debugPublisher.publish(str(pred))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)

    orb = NetworkTestClient(
        f"/home/mike/dev/orb_imitation/models/ResNet8_ds=dr_pretrain_l={itypes}_f=0.5_bs=32_lt=MSE_lr=0.001_c=run0/best.pth")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Stopping node")
