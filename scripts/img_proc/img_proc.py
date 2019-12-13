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

# Launch file created to attend the requirements established on the Ex4 by the discipline of Intelligent Control \
#      of Robotics Systems
# Professor: Wouter Caarls
# Students: Matheus do Nascimento Santos 1920858  (@matheusns)
#           Luciana Reys 1920856 (@lsnreys)

import cv2
import numpy as np

def process_img(img):
    dst = img.copy()
    # Extract HSV channels
    hsv_image = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    # Boundaries definition
    lower = np.array([81, 150, 8])
    upper = np.array([111, 255, 140])

    # Final mask
    mask = cv2.inRange(hsv_image, lower, upper)

    # Invoke morphological process
    gradient = morphological_process(mask)

    filled_img = img_fill(gradient)

    return filled_img, hsv_image

def morphological_process(src):
    # dst = src.copy()
    # # Ensure Threshold is applied
    # dst[:] = np.where(dst > 0, 255, 0)
    # kernel = np.ones((15, 15), np.uint8)
    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))
    maskOpen = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    # resized_image = cv2.resize(maskClose, (320, 180))
    # cv2.namedWindow("self.ADJUSTMENT_WINDOW", cv2.WINDOW_NORMAL)
    # cv2.imshow("self.ADJUSTMENT_WINDOW", resized_image)
    # key = cv2.waitKey(1)

    return maskClose

def img_fill(src, debug=False):
    # Copy the thresholded image.
    im_floodfill = src.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = src.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = src | im_floodfill_inv

    if (debug):
        temp = np.vstack([np.hstack([src, fill_image])])
        resized_image = cv2.resize(temp, (320, 180))
        cv2.namedWindow("bla", cv2.WINDOW_NORMAL)
        cv2.imshow("bla", resized_image)
        key = cv2.waitKey(1)

    return fill_image

def findContours(src, rgb):
    (im2, contours, hierarchy) = cv2.findContours(src.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height = src.shape[0]
    width = src.shape[1]
    min_x, min_y = width, height
    max_x = max_y = 0
    dst = src.copy()
    possible_contour = None
    roi = None
    rgb_bounded = None

    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)

        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)

        possible_contour = contour
        roi = src[y:y+h, x:x+w]

    return possible_contour

def plot_bounding_box(src, rgb):
    _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
    else:
        x, y, w, h = cv2.boundingRect(src)
    box_coordinates = (x,y,w,h)
    color = (0, 0, 255)
    rgb_bounded = rgb.copy()
    tmp = cv2.rectangle(rgb_bounded, (x, y), (x + w, y + h), color, 3)

    return rgb_bounded, box_coordinates
