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

#OpenCV modules
import cv2

#Other modules
import numpy as np


# Process image main function
def process_img(img):
    dst = img.copy()
    # Extract HSV channels
    hsv_image = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    # Boundaries definition
    lower = np.array([0, 68, 65])
    upper = np.array([11, 255, 208])

    # Final mask
    mask = cv2.inRange(hsv_image, lower, upper)

    # smooth to reduce noise a bit more
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Invoke morphological process
    gradient = morphological_process(blurred)

    # Apply flood fill algorithm
    filled_img = img_fill(gradient)

    return filled_img


# Applies the open and close morphological algorithms
def morphological_process(src):
    kernel_open = np.ones((5, 5))
    kernel_close = np.ones((5, 5))
    mask_open = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    return mask_close


# Fills images using Flood fill algorithm
def img_fill(src, debug=False):
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

    return fill_image


# Plots the bounding box into the rgb image according to the desired approach
def plot_bounding_box(src, rgb):
    _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        contour = upper_left_contour(contours)
        x, y, w, h = cv2.boundingRect(contour)
    else:
        x, y, w, h = cv2.boundingRect(src)
    box_coordinates = (x, y, w, h)
    color = (0, 0, 255)
    rgb_bounded = rgb.copy()
    tmp = cv2.rectangle(rgb_bounded, (x, y), (x + w, y + h), color, 3)

    return rgb_bounded, box_coordinates


# Defines the most upper left bounding box
def upper_left_contour(contours):
    sorted_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        sorted_contours.append((x, y, cnt))

    sorted_contours.sort(key=lambda tup: tup[0])

    return sorted_contours[0][2]
