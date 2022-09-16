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

import UnityPID as PID
import util as util

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

        self.ctrl = PID.VelocityPID(Kp, Ki, Kd, yaw_gain)

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

    def updatePID(self) -> None:

        if self.pose is not None:
            # update pid controller
            _, _, WposeYaw = util.to_eularian_angles(self.pose.pose.orientation.w, self.pose.pose.orientation.x,
                                                     self.pose.pose.orientation.y, self.pose.pose.orientation.z)
            Wpose = [self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z, WposeYaw]

            # update pid controller
            self.ctrl.setState(Wpose)
            self.ctrl.update(rospy.Time.now() - self.lastUpdate)
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
