"""Main file of the robotic arm project."""

# Import python libraries
import sys
import time
import logging
import datetime

# Import external python packages for object detection
import numpy as np
import cv2 as cv

# Import relevant methods
from stack_images import stack_images

# Import the robotic arm and the camera class
from camera import Camera

def nothing(x):
    pass

# Initialize constants to exit the program
WAIT_KEYPRESS_MSEC: int = 20
MODIFIER_MASK: int = 0xFF
ESC_KEY: int = 27


# Variable to determine if it is the first loop
FIRST_LOOP: bool = True

# Variable to determine if objects and when they are found
OBJECTS_FOUND_AT: int = 0
OBJECTS_FOUND: bool = False

# Create window to set parameters
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
# Set thresholds for Canny edge detection
# Upper threshold
cv.createTrackbar("Threshold1", "Parameters", 40, 255, lambda _: ...)
# Lower threshold
cv.createTrackbar("Threshold2", "Parameters", 120, 255, lambda _: ...)
# Minimum area required for an object to be detected
cv.createTrackbar("Area (min)", "Parameters", 1500, 100000, lambda _: ...)
# Maximum area for an object to be detected
cv.createTrackbar("Area (max)", "Parameters", 7000, 100000, lambda _: ...)

# Create a window
cv.namedWindow('image')
# create trackbars for color change
cv.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv.createTrackbar('SMin','image',0,255,nothing)
cv.createTrackbar('VMin','image',0,255,nothing)
cv.createTrackbar('HMax','image',0,179,nothing)
cv.createTrackbar('SMax','image',0,255,nothing)
cv.createTrackbar('VMax','image',0,255,nothing)
# Set default value for MAX HSV trackbars.
cv.setTrackbarPos('HMax', 'image', 179)
cv.setTrackbarPos('SMax', 'image', 105)
cv.setTrackbarPos('VMax', 'image', 255)

while True:
    # Give the camera a frame to correctly initialize
    if FIRST_LOOP:
        FIRST_LOOP = False
        continue
    
    image_original = cv.imread('Picture_45.jpg')
    gray_median_frame = cv.imread('Picture_0.jpg')
    gray_median_frame = cv.cvtColor(gray_median_frame, cv.COLOR_BGR2GRAY)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    waitTime = 33

    # get current positions of all trackbars
    hMin = cv.getTrackbarPos('HMin','image')
    sMin = cv.getTrackbarPos('SMin','image')
    vMin = cv.getTrackbarPos('VMin','image')

    hMax = cv.getTrackbarPos('HMax','image')
    sMax = cv.getTrackbarPos('SMax','image')
    vMax = cv.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv.cvtColor(image_original, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    image_hsv = cv.bitwise_and(image_original,image_original, mask= mask)
    cv.imshow('hsv',image_hsv)

    # Blur the image to better detect edges
    image_blurred: np.ndarray[np.uint8] = cv.GaussianBlur(image_hsv, (7, 7), 1)
    # Convert image to grayscale to better detect edges
    image_grayscale: np.ndarray[np.uint8] = cv.cvtColor(
        image_blurred, cv.COLOR_BGR2GRAY)
    # Calculate absolute difference of current image and median frame
    image_difference: np.ndarray[np.uint8] = cv.absdiff(
        image_grayscale, gray_median_frame)
    # Creating a mask to stop detection from outside the area
    mask=np.zeros(image_difference.shape[:2], dtype="uint8")
    cv.circle(mask,(325,0),446,255,-1)
    image_masked: np.ndarray[np.uint8]=cv.bitwise_and(
        image_difference, image_difference,mask=mask)
    cv.imshow('mask',image_masked)
    # Threshold to binarize
    _threshold: float
    image_difference: np.ndarray[np.uint8]
    _threshold, image_masked = cv.threshold(
        image_masked, 30, 255, cv.THRESH_BINARY)
    # Get threshold values from trackbar
    threshold1: int = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2: int = cv.getTrackbarPos("Threshold2", "Parameters")
    # Detect edges using Canny
    image_canny: np.ndarray[np.uint8] = cv.Canny(
        image_masked, threshold1, threshold2)
    # Remove overlaps and noise using dilation
    kernel: np.ndarray[np.float64] = np.ones((5, 5))
    image_dilation: np.ndarray[np.uint8] = cv.dilate(
        image_canny, kernel, iterations=1)
    # Get contours of image to detect objects
    image_objects: np.ndarray[np.uint8] = image_original.copy()
    centroids: list[tuple[int, int]] = Camera.get_contours(
        image_dilation, image_objects)

    # Put text on each image to identify them
    cv.putText(image_original, "Original Image", (10, 25),
            cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_difference, "Background removed", (10, 25),
            cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_dilation, "Edge detection using Canny", (10, 25),
            cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_objects, "Objects detected", (10, 25),
            cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    # Stack the images into one window
    image_stack: np.ndarray[np.uint8]
    image_stack = stack_images(0.8, ([image_original, image_difference],
                                    [image_dilation, image_objects]))
    # Display each frame
    cv.imshow("Result", image_stack)

    OBJECTS_FOUND = False
    OBJECTS_FOUND_AT = 0
    FIRST_LOOP = True

    # Determine the script by pressing the ESC-key
    if cv.waitKey(WAIT_KEYPRESS_MSEC) & MODIFIER_MASK == ESC_KEY:
        break

cv.destroyAllWindows()