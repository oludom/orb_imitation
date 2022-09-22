#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import message_filters as mf

import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import time

# nice terminal output
import curses


class OrbExtractor:

    def __init__(self) -> None:
        self.firstCall = True
        self.c = curses.initscr()

        self.bridge = CvBridge()
        lisub = mf.Subscriber('/zedm/zed_node/left/image_rect_color', Image)
        risub = mf.Subscriber("/zedm/zed_node/right/image_rect_color", Image)

        self.ts = mf.TimeSynchronizer([lisub, risub], 10)
        self.ts.registerCallback(self.callback)

        self.debugPublisher = rospy.Publisher("/debugImage", Image, queue_size=10)

        self.orb = cv.ORB_create(nfeatures=1500)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def callback(self, limsg, rimsg) -> None:

        # if self.firstCall:
        # rospy.loginfo("Camera connected, receiving images.")
        # self.firstCall = False
        self.c.addstr(0, 0, "Camera connected, receiving images. " + str(time.time()))

        orb = self.orb
        m = self.matcher

        try:
            # convert image
            lcvImage = self.bridge.imgmsg_to_cv2(limsg, "bgr8")
            rcvImage = self.bridge.imgmsg_to_cv2(rimsg, "bgr8")

            # detect orb features
            lkp = orb.detect(lcvImage, None)
            rkp = orb.detect(rcvImage, None)

            # compute descriptors
            lkp2, ldesc = orb.compute(lcvImage, lkp)
            rkp2, rdesc = orb.compute(rcvImage, rkp)

            # check if descriptors are available and have the same shape
            if ldesc is not None and rdesc is not None and ldesc.shape[1] == rdesc.shape[1]:
                # match descriptors of left and right image
                matches = m.match(ldesc, rdesc)

                matches = list(filter(lambda m: self.lineBetweenPointsHorizontal(m, lkp2, rkp2), matches))
                self.c.addstr(1, 0, f"keypoints: {str(len(lkp2)).zfill(4)} - {str(len(rkp2)).zfill(4)}")
                self.c.addstr(2, 0, f"matches: {str(len(matches))}")

                # sort matches based on distance
                matches = sorted(matches, key=lambda x: x.distance)

                # draw matches on debug image
                debugImage = cv.drawMatches(lcvImage, lkp2, rcvImage, rkp2, matches, None)

                # convert and publish debug image
                debugMsg = self.bridge.cv2_to_imgmsg(debugImage, "bgr8")
                self.debugPublisher.publish(debugMsg)
        except CvBridgeError as e:
            print(e)

        self.c.refresh()

    def lineBetweenPointsHorizontal(self, match, keypoints1, keypoints2) -> bool:

        idx1 = match.queryIdx
        idx2 = match.trainIdx

        l1 = len(keypoints1)
        l2 = len(keypoints2)

        if not idx1 >= 0 and not l1 > idx1 and not idx2 >= 0 and not l2 > idx2:
            return False

        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        if kp1 and kp2 and kp1.pt and kp2.pt:
            x1, y1 = kp1.pt
            x2, y2 = kp2.pt

            if (x2 - x1) == 0:
                return False

            # calculate slope between the two key points
            slope = np.abs((y2 - y1) / (x2 - x1))

            if slope > .1:
                return False
            else:
                return True
        else:
            return False


if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)

    orb = OrbExtractor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Stopping node")

    curses.endwin()
