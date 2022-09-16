#!/usr/bin/env python
import rospy
# from sensor_msgs.msg import Image
# import message_filters as mf

import numpy as np
# import cv2 as cv
# from cv_bridge import CvBridge, CvBridgeError
# import time

# nice terminal output
# import curses


#!/usr/bin/env python

import time
import numpy as np
from math import *

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
        print(f"target: {targetAngle}, current: {currentAngle}, error: {error}")

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






from geometry_msgs.msg import PoseStamped, TwistStamped


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
        self.ctrl.setGoal([2., 2., 2., 0.])

        # subscribe to pose
        self.pose_sub = rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.callback)

        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.updatePID)

        # self.c = curses.initscr()

        # self.bridge = CvBridge()
        # lisub = mf.Subscriber('/zedm/zed_node/left/image_rect_color', Image)
        # risub = mf.Subscriber("/zedm/zed_node/right/image_rect_color", Image)
        #
        # self.ts = mf.TimeSynchronizer([lisub, risub], 10)
        # self.ts.registerCallback(self.callback)
        #
        # self.debugPublisher = rospy.Publisher("/debugImage", Image, queue_size=10)
        #
        # self.orb = cv.ORB_create(nfeatures=1500)
        # self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def callback(self, pose: PoseStamped) -> None:

        if self.firstCall:
            rospy.loginfo("Connected, receiving pose.")
            self.firstCall = False

        self.pose = pose

    def updatePID(self, *_) -> None:

        if self.pose is not None:
            # update pid controller
            _, _, WposeYaw = to_eularian_angles(self.pose.pose.orientation.w, self.pose.pose.orientation.x,
                                                     self.pose.pose.orientation.y, self.pose.pose.orientation.z)
            Wpose = [self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z, WposeYaw]

            # update pid controller
            self.ctrl.setState(Wpose)
            dt = rospy.Time.now() - self.lastUpdate
            print("dt: ", dt.to_sec())
            self.ctrl.update(dt.to_sec())
            # get current pid output´
            Wvel, Wyaw = self.ctrl.getVelocityYaw()

            self.lastUpdate = rospy.Time.now()

            # publish velocity
            vel = TwistStamped()
            vel.header.stamp = rospy.Time.now()
            vel.twist.linear.x = Wvel[0]
            vel.twist.linear.y = Wvel[1]
            vel.twist.linear.z = Wvel[2]
            vel.twist.angular.z = Wyaw
            self.vel_pub.publish(vel)


if __name__ == '__main__':

    rospy.init_node('pid_controller', anonymous=True)

    vcn = VelocityControllerNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Stopping node")

    # curses.endwin()
