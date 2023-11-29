import sys
import cv2 as cv
import numpy as np
from robotarm.robot.robotic_arm import RoboticArm
from robotarm.camera.camera import Camera

arm: RoboticArm = RoboticArm()

# Initialize constants to exit the program
WAIT_KEYPRESS_MSEC: int = 20
MODIFIER_MASK: int = 0xFF
ESC_KEY: int = 27

while True:
    cam: Camera = Camera(640, 480, cv.VideoCapture(0))
    success: bool
    image_original: np.ndarray[np.uint8]
    success, image_original = cam.capture.read()

    # Blur the image to better detect edges
    image_blurred: np.ndarray[np.uint8] = cv.GaussianBlur(image_original, (7, 7), 1)

    # Convert image to grayscale to better detect edges
    image_grayscale: np.ndarray[np.uint8] = cv.cvtColor(
        image_blurred, cv.COLOR_BGR2GRAY)

    # Creating a mask to stop detection from outside the area
    mask=np.zeros(image_grayscale.shape[:2], dtype="uint8")
    cv.circle(mask,(325,7),434,255,-1)
    image_masked: np.ndarray[np.uint8]=cv.bitwise_and(
        image_grayscale, image_grayscale,mask=mask)

    # Threshold to binarize
    _threshold: float
    image_grayscale: np.ndarray[np.uint8]
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
    
    for x in centroids:
        print(x)

    # Determine the script by pressing the ESC-key
    if cv.waitKey(WAIT_KEYPRESS_MSEC) & MODIFIER_MASK == ESC_KEY:
        arm.go_to_starting_position()
        break
