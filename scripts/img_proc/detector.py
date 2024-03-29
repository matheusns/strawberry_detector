#!/usr/bin/env python
# BSD 3-Clause License
#
# Copyright (c) 2019, Matheus Nascimento, Luciana Reys
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Professor: Wouter Caarls
# Students: Matheus do Nascimento Santos 1920858  (@matheusns)
#           Luciana Reys 1920856 (@lsnreys)

# OpenCV modules
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Ros modules
import rospy
from sensor_msgs.msg import Image, JointState, CameraInfo
from strawberry_detector.srv import StwPose, StwPoseResponse

# Other modules
import sys
import numpy as np
import img_proc


def nothing(x):
    pass


class StrawberryDetector:
    def __init__(self):
        rospy.init_node('img_proc_node')
        self.init_members_variables()
        self.init_ros_channels()
        # rospy.spin()
        self.img_debug()
        # self.calibrate_img()

    def init_members_variables(self):
        self.debug = False
        self.bridge = CvBridge()
        self.cv_image = None

        self.MINIMUM_HUE = "MINIMUM_HUE"
        self.MAXIMUM_HUE = "MAXIMUM_HUE"
        self.MINIMUM_SATURATION = "MINIMUM_SATURATION"
        self.MAXIMUM_SATURATION = "MAXIMUM_SATURATION"
        self.MINIMUM_VALUE = "MINIMUM_VALUE"
        self.MAXIMUM_VALUE = "MAXIMUM_VALUE"
        self.ADJUSTMENT_WINDOW = "FINAL OUTPUT"

    def init_ros_channels(self):
        self.image_sub = rospy.Subscriber("/img_acquistion/camera/image", Image, self.callback, queue_size=1)
        self.service_server = rospy.Service('img_proc', StwPose, self.detect_strawberries)
        # todo Add camera info structure
        # self.camera_info_sub = rospy.Subscriber('gpg/camera_info', CameraInfo, self.cameraInfoCallback)

    def callback(self, msg):
        rospy.logdebug("Received an image!")
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if (self.debug):
                print "Here"
                cv2.namedWindow(self.ADJUSTMENT_WINDOW, cv2.WINDOW_NORMAL)
                cv2.imshow(self.ADJUSTMENT_WINDOW, self.cv_image)
                cv2.waitKey(1)

        except CvBridgeError, e:
            print(e)

    # todo Add camera info structure
    # def cameraInfoCallback(self, msg):
    #     self.model.fromCameraInfo(msg)

    def calibrate_img(self):
        rate = rospy.Rate(30)

        cv2.namedWindow(self.ADJUSTMENT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.MINIMUM_HUE, self.ADJUSTMENT_WINDOW, 0, 255, nothing)
        cv2.createTrackbar(self.MAXIMUM_HUE, self.ADJUSTMENT_WINDOW, 0, 255, nothing)
        cv2.createTrackbar(self.MINIMUM_SATURATION, self.ADJUSTMENT_WINDOW, 0, 255, nothing)
        cv2.createTrackbar(self.MAXIMUM_SATURATION, self.ADJUSTMENT_WINDOW, 0, 255, nothing)
        cv2.createTrackbar(self.MINIMUM_VALUE, self.ADJUSTMENT_WINDOW, 0, 255, nothing)
        cv2.createTrackbar(self.MAXIMUM_VALUE, self.ADJUSTMENT_WINDOW, 0, 255, nothing)

        while not rospy.is_shutdown():
            rospy.wait_for_message("/img_acquistion/camera/image", Image)
            hMin = cv2.getTrackbarPos(self.MINIMUM_HUE, self.ADJUSTMENT_WINDOW)
            hMax = cv2.getTrackbarPos(self.MAXIMUM_HUE, self.ADJUSTMENT_WINDOW)
            sMin = cv2.getTrackbarPos(self.MINIMUM_SATURATION, self.ADJUSTMENT_WINDOW)
            sMax = cv2.getTrackbarPos(self.MAXIMUM_SATURATION, self.ADJUSTMENT_WINDOW)
            vMin = cv2.getTrackbarPos(self.MINIMUM_VALUE, self.ADJUSTMENT_WINDOW)
            vMax = cv2.getTrackbarPos(self.MAXIMUM_VALUE, self.ADJUSTMENT_WINDOW)

            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            if (self.debug):
                print lower
                print "###############"
                print upper

            cv2.namedWindow(self.ADJUSTMENT_WINDOW, cv2.WINDOW_NORMAL)
            hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            temp = np.vstack([np.hstack([self.cv_image, hsv_image, mask])])
            resized_image = cv2.resize(temp, (640, 360))
            cv2.imshow(self.ADJUSTMENT_WINDOW, resized_image)
            key = cv2.waitKey(1)

            rate.sleep()

    def detect_strawberries(self, req):
        rate = rospy.Rate(15)
        count = 0
        while not rospy.is_shutdown():
            current_img = self.cv_image
            if current_img is not None:
                segmented_img = img_proc.process_img(current_img)
                output, boundaries, center = img_proc.plot_bounding_box(segmented_img, current_img)
                return StwPoseResponse(boundaries[0], boundaries[1])

    def img_debug(self):
            rate = rospy.Rate(15)
            count = 0
            while not rospy.is_shutdown():
                current_img = self.cv_image
                if current_img is not None:
                    segmented_img = img_proc.process_img(current_img)
                    output, boundaries, _ = img_proc.plot_bounding_box(segmented_img, current_img)

                    cv2.namedWindow(self.ADJUSTMENT_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.imshow(self.ADJUSTMENT_WINDOW, output)
                    key = cv2.waitKey(1)
                    rate.sleep()


def main(args):
    main_app = StrawberryDetector()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
