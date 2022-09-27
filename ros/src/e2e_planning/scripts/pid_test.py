#!/usr/bin/env python3
import rospy

import numpy as np


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
resnet_factor = 0.5
num_train_tracks = 170
num_val_tracks = 50
jobs = 8

input_channels = {
    'rgb': True,
    'depth': False,
    'orb': False
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










'''
PID controller based on 
https://vazgriz.com/621/pid-controllers/
'''


class PIDController:
    # pg: p-gain
    # ig: i-gain
    # dg: d-gain
    def __init__(self, pg, ig, dg):
        self.proportionalGain = pg
        self.integralGain = ig
        self.derivativeGain = dg

        self.valueLast = 0
        self.integrationStored = 0

        self.errorOutput = ""

    # use for linear values e.g. x, y, z
    def update(self, dt: float, currentValue: float, targetValue: float):
        error = targetValue - currentValue

        P = self.proportionalGain * error

        self.integrationStored = self.integrationStored + (error * dt)
        I = self.integralGain * self.integrationStored
        # I = max(min(I, 5), -5)

        valueRateOfChange = (currentValue - self.valueLast) / dt
        self.valueLast = currentValue

        D = self.derivativeGain * -valueRateOfChange

        self.errorOutput = f"Pe: {error}, De: {-valueRateOfChange}"

        return P + I + D

    # all angles in degree
    def angleDifference(self, a: float, b: float):
        return (a - b + 540) % 360 - 180

    # use for angle values e.g. yaw
    def updateAngle(self, dt: float, currentAngle: float, targetAngle: float):
        error = self.angleDifference(targetAngle, currentAngle)
        # print(f"target: {targetAngle}, current: {currentAngle}, error: {error}")

        P = self.proportionalGain * error

        self.integrationStored = self.integrationStored + (error * dt)
        I = self.integralGain * self.integrationStored
        I = max(min(I, 2), -2)

        valueRateOfChange = self.angleDifference(currentAngle, self.valueLast) / dt
        self.valueLast = currentAngle

        D = self.derivativeGain * -valueRateOfChange

        self.errorOutput = f"Pe: {error}, De: {-valueRateOfChange}"

        return max(min(P + I + D, self.yaw_limit), -self.yaw_limit)


'''
velocity pid controller
outputs a velocity command in world frame, 
based on input x, y, z, and yaw in world frame

'''


class VelocityPID:
    '''
    pid controller
    kp: np.array 3, p gain for x, y, z
    ki: np.array 3, i gain for x, y, z
    kd: np.array 3, d gain for x, y, z
    yg: np.array 3, yaw gains, p, i, d
    dthresh: float, distance threshold
    athresh: float, angle threshold

    '''

    def __init__(self, kp=np.array([0., 0., 0.]), ki=np.array([0., 0., 0.]), kd=np.array([0., 0., 0.]), yg=0.,
                 dthresh=.1, athresh=.1):
        # in degrees, max requested change
        self.yaw_limit = 10

        # state
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0

        # previous
        self.previous_distance_error = np.array([0, 0, 0])
        self.previous_angle_error = np.array([0, 0, 0])
        self.previous_integral_error = np.array([0, 0, 0])

        # Default state
        self.x_goal = 0
        self.y_goal = 0
        self.z_goal = 0
        self.yaw_goal = 0

        # gains
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.yaw_gain = yg

        # output values
        self.velocity_out = np.array([0, 0, 0])
        self.yaw_out = 0

        # error publisher
        self.errorOutput = ""
        self.errorOutput1 = ""
        self.errorOutput2 = ""
        self.errorOutput3 = ""

        # PID controller objects
        self.cx = PIDController(kp[0], ki[0], kd[0])
        self.cy = PIDController(kp[1], ki[1], kd[1])
        self.cz = PIDController(kp[2], ki[2], kd[2])
        self.cyaw = PIDController(yg[0], yg[1], yg[2])

    # [x, y, z, yaw]
    def setState(self, state):
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.yaw = state[3]

    # [x, y, z, yaw]
    def setGoal(self, state):
        self.x_goal = state[0]
        self.y_goal = state[1]
        self.z_goal = state[2]
        self.yaw_goal = state[3]

    # returns output of pid controller
    def getVelocityYaw(self):
        return (self.velocity_out, self.yaw_out)

    # df: time delta since last update
    def update(self, dt):
        # Retrieve the UAV's state
        x = self.x
        y = self.y
        z = self.z
        yaw = self.yaw

        goal_x = self.x_goal
        goal_y = self.y_goal
        goal_z = self.z_goal

        # set current output values
        self.velocity_out = np.array(
            [self.cx.update(dt, x, goal_x), self.cy.update(dt, y, goal_y), self.cz.update(dt, z, goal_z)])
        self.yaw_out = self.cyaw.update(dt, yaw, self.yaw_goal)
        self.yaw_out = max(min(self.yaw_out, self.yaw_limit), -self.yaw_limit)

        # error output
        self.errorOutput = ""
        self.errorOutput1 = f"x: {self.cx.errorOutput}"
        self.errorOutput2 = f"y: {self.cy.errorOutput}"
        self.errorOutput3 = f"z: {self.cz.errorOutput}"


import math

import numpy as np
import math as m


# calculate magnitude of vector
def magnitude(vec):
    return np.sqrt(vec.dot(vec))


# theta: angle in radian
def Rx(theta: float):
    return np.array([[1, 0, 0],
                     [0, m.cos(theta), m.sin(theta)],
                     [0, -m.sin(theta), m.cos(theta)]])


# theta: angle in radian
def RxT(theta: float):
    return Rx(theta).T


# theta: angle in radian
def Ry(theta: float):
    return np.array([[m.cos(theta), 0, -m.sin(theta)],
                     [0, 1, 0],
                     [m.sin(theta), 0, m.cos(theta)]])


# theta: angle in radian
def RyT(theta: float):
    return Ry(theta).T


# theta: angle in radian
def Rz(theta: float):
    return np.array([[m.cos(theta), m.sin(theta), 0],
                     [-m.sin(theta), m.cos(theta), 0],
                     [0, 0, 1]])


# theta: angle in radian
def RzT(theta: float):
    return Rz(theta).T


# convert vector from world to body frame
# v: vector to transform in world frame
# b: position of body in world frame / translation offset
# by: body yaw in radian
def vector_world_to_body(v: np.ndarray, b: np.ndarray, by: float) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return Rz(by) @ (v - b)


# convert vector from body to world frame
# v: vector to transform in body frame
# b: position of body in world frame / translation offset
# by: body yaw in radian
def vector_body_to_world(v: np.ndarray, b: np.ndarray, by: float) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return (RzT(by) @ v) + b


# helper method for converting getOrientation to roll/pitch/yaw
# https:#en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

def to_eularian_angles(w, x, y, z):
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return pitch, roll, yaw

def quat2eul(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    X = math.atan2(t0, t1)
    Y = math.asin(t2)
    Z = math.atan2(t3, t4)
    return np.array([X, Y, Z])


def to_quaternion(pitch, roll, yaw):
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    w = t0 * t2 * t4 + t1 * t3 * t5  # w
    x = t0 * t3 * t4 - t1 * t2 * t5  # x
    y = t0 * t2 * t5 + t1 * t3 * t4  # y
    z = t1 * t2 * t4 - t0 * t3 * t5  # z
    return w, x, y, z


from math import degrees, radians

from geometry_msgs.msg import PoseStamped, TwistStamped

from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import SetMode, SetModeRequest, SetModeResponse, CommandBool, CommandBoolRequest, \
    CommandBoolResponse, CommandTOL, CommandTOLRequest


class VelocityControllerNode:

    def __init__(self) -> None:
        self.firstCall = True
        self.lastUpdate = rospy.Time.now()
        self.pose = None

        # init pid controller for velocity control
        pp = 2
        dd = .2
        ii = .0

        Kp = np.array([pp, pp, pp])
        Ki = np.array([ii, ii, ii])
        Kd = np.array([dd, dd, dd])
        yaw_gain = np.array([.2, 0., 25])  # [.25, 0, 0.25]

        self.ctrl = VelocityPID(Kp, Ki, Kd, yaw_gain)

        # set initial state and goal to 0
        self.ctrl.setState([0, 0, 0, 0])
        self.ctrl.setGoal([0, 0, 0, 0])
        # self.ctrl.setGoal([2., 2., 2., 0.])

        # subscribe to pose
        self.pose_sub = rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.callback)
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.update)
        self.current_state = State()
        rospy.Subscriber('/mavros/state', State, self._current_state_cb)

        self.set_mode("OFFBOARD")

    def _current_state_cb(self, data):
        self.current_state = data

    def callback(self, pose: PoseStamped) -> None:

        if self.firstCall:
            rospy.loginfo("Connected, receiving pose.")
            self.firstCall = False

        self.pose = pose

    # pose: x,y,z,yaw in degrees
    def getState(self):
        _, _, WposeYaw = to_eularian_angles(self.pose.pose.orientation.w, self.pose.pose.orientation.x,
                                            self.pose.pose.orientation.y, self.pose.pose.orientation.z)
        return [self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z, WposeYaw]

    def angleDifference(self, a: float, b: float):
        return (a - b + 540) % 360 - 180

    def update(self, *_) -> None:

        wp = [2., 2., 2., 0.]

        if self.pose is not None:
            # update pid controller

            # get current state
            Wcstate = self.getState()

            # set goal state of pid controller
            Bgoal = vector_world_to_body(wp[:3], Wcstate[:3], Wcstate[3])
            # desired yaw angle is target point yaw angle world minus current uav yaw angle world
            ByawGoal = self.angleDifference(wp[3], degrees(Wcstate[3]))
            print(f"angle target: {ByawGoal:5.4f}")
            self.ctrl.setGoal([*Bgoal, ByawGoal])

            # update pid controller
            dt = rospy.Time.now() - self.lastUpdate
            print("dt: ", dt.to_sec())
            self.ctrl.update(dt.to_sec())

            # get current pid output´
            Bvel, Byaw = self.ctrl.getVelocityYaw()

            print(f"magnitude: {magnitude(Bvel)}")
            Bvel_percent = magnitude(Bvel) / 2
            print(f"percent: {Bvel_percent * 100}")
            # if magnitude of pid output is greater than velocity limit, scale pid output to velocity limit
            if Bvel_percent > 1:
                Bvel = Bvel / Bvel_percent

            # rotate velocity command such that it is in world coordinates
            Wvel = vector_body_to_world(Bvel, [0, 0, 0], Wcstate[3])

            # add pid output for yaw to current yaw position
            # Wyaw = Wcstate[3] + radians(Byaw)

            self.lastUpdate = rospy.Time.now()

            # publish velocity
            vel = TwistStamped()
            vel.header.stamp = rospy.Time.now()
            vel.twist.linear.x = Wvel[0]
            vel.twist.linear.y = Wvel[1]
            vel.twist.linear.z = Wvel[2]
            vel.twist.angular.z = radians(Byaw)
            self.vel_pub.publish(vel)

    ### service function ###

    def set_mode(self, mode):
        # if not self.current_state.connected:
        #     print(
        #     "No FCU connection")
        #
        # elif self.current_state.mode == mode:
        #     print
        #     "Already in " + mode + " mode"
        #
        # else:

        # wait for service
        rospy.wait_for_service("mavros/set_mode")

        # service client
        set_mode = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # set request object
        req = SetModeRequest()
        req.custom_mode = mode

        # zero time
        t0 = rospy.get_time()

        # check response
        while not rospy.is_shutdown() and (self.current_state.mode != req.custom_mode):
            if rospy.get_time() - t0 > 2.0:  # check every 5 seconds

                try:
                    # request
                    set_mode.call(req)

                except rospy.ServiceException as e:
                    print(f"Service did not process request: {str(e)}")

                t0 = rospy.get_time()

        print("Mode: " + self.current_state.mode + " established")


class ResnetControllerNode (VelocityControllerNode):

    def __init__(self, modelPath, raceTrackName="track0", device='cuda', *args, **kwargs):
        super().__init__()

        self.loaded = False

        # ros setup
        self.firstCall = True

        self.bridge = CvBridge()
        lisub = mf.Subscriber('/zedm/zed_node/left/image_rect_color', Image)
        risub = mf.Subscriber("/zedm/zed_node/depth/depth_registered", Image)

        self.ts = mf.TimeSynchronizer([lisub, risub], 1)
        self.ts.registerCallback(self.updateNetwork)

        # self.debugPublisher = rospy.Publisher("/vel", String, queue_size=10)

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

        self.sequence = 0


    def preprocessImages(self, image, depth):

        sample = None
        withDepth = input_channels['depth']

        if input_channels['orb']:
            kp, des, _, _, _ = get_orb(image)

        # preprocess image
        image = transforms.Compose([
            transforms.ToTensor()
        ])(image)

        if input_channels['rgb']:
            sample = image

        if withDepth:
            depth = transforms.Compose([
                transforms.ToTensor()
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

    def updateNetwork(self, limsg, depthmsg) -> None:

        if not self.loaded or self.pose is None:
            return
        if self.firstCall:
            rospy.loginfo("Camera connected, receiving images.")
            self.firstCall = False

        sn = self.sequence
        self.sequence += 1
        tn = now()

        try:
            # convert image
            if input_channels['rgb'] or input_channels['orb']:
                lcvImage = np.frombuffer(limsg.data, dtype=np.uint8).reshape(limsg.height, limsg.width, -1)
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

            pd(tn, f"{sn} preprocessing")

            # predict vector with network
            pred = self.model(sample)
            pred = pred.to(torch.device('cpu'))
            pred = pred.detach().numpy()
            pred = pred[0]  # remove batch

            pd(tn, f"{sn} prediction")

            Bvel, Byaw = pred[0:3], pred[3]
            Byaw *= 10

            # get current state
            Wcstate = self.getState()


            # print(f"magnitude: {magnitude(Bvel)}")
            Bvel_percent = magnitude(Bvel) / .5
            # print(f"percent: {Bvel_percent * 100}")
            # if magnitude of pid output is greater than velocity limit, scale pid output to velocity limit
            if Bvel_percent > 1:
                Bvel = Bvel / Bvel_percent

            # rotate velocity command such that it is in world coordinates
            Wvel = vector_body_to_world(Bvel, [0, 0, 0], Wcstate[3])

            # publish velocity
            vel = TwistStamped()
            vel.header.stamp = rospy.Time.now()
            vel.twist.linear.x = Wvel[0]
            vel.twist.linear.y = Wvel[1]
            vel.twist.linear.z = Wvel[2]
            vel.twist.angular.z = radians(-Byaw)
            self.vel_pub.publish(vel)

        except CvBridgeError as e:
            print(e)
        except Exception as e:
            # publish velocity
            vel = TwistStamped()
            vel.header.stamp = rospy.Time.now()
            vel.twist.linear.x = 0.
            vel.twist.linear.y = 0.
            vel.twist.linear.z = 0.
            vel.twist.angular.z = 0.
            self.vel_pub.publish(vel)
            print(e)


if __name__ == '__main__':

    rospy.init_node('pid_controller', anonymous=True)

    # vcn = VelocityControllerNode()
    vcn = ResnetControllerNode(
        f"/home/kakao/micha/orb_imitation/models/ResNet8_ds=dr_pretrain_l={itypes}_f=0.5_bs=32_lt=MSE_lr=0.001_c=run0/best.pth")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Stopping node")
