"""Module that currently acts as the main-file of the camera class/package."""

# Import necessary libraries
import sys
import cv2 as cv
import numpy as np
import logging
import datetime

# Import other files
from robotarm.camera.camera import Camera
from robotarm.camera.stack_images import stack_images

# Initialize the logger
logger = logging.getLogger("__video__")
logger.setLevel(logging.INFO)

# Initialize the handler
stdout = logging.StreamHandler(stream=sys.stdout)
stdout.setLevel(logging.INFO)
currentTime = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
filename = 'home/pi/Dokumente/Robotic_arm_newest/RoboArm-main/roboarm/src/logs/runtime:' + currentTime + '.log'
fileout = logging.FileHandler(filename)
fileout.setLevel(logging.INFO)

# Initialize the formatter
fmt = logging.Formatter("%(name)-11s : %(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)s >>> %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Set formatter
stdout.setFormatter(fmt)
fileout.setFormatter(fmt)

# Add handler
logger.addHandler(stdout)
logger.addHandler(fileout)

# Initialize constants to exit the program
WAIT_KEYPRESS_MSEC: int = 20
MODIFIER_MASK: int = 0xFF
ESC_KEY: int = 27

# Initialize camera settings
camera_one: Camera = Camera(640, 480, cv.VideoCapture(0))

# Check if capturing works
if not camera_one.capture.isOpened():
    logger.error("Cannot open camera")
    sys.exit()

# Create window to set parameters
cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
# Set thresholds for Canny edge detection
# Upper threshold
cv.createTrackbar("Threshold1", "Parameters", 40, 255, lambda _: ...)
# Lower Threshold
cv.createTrackbar("Threshold2", "Parameters", 120, 255, lambda _: ...)
# Minimum area required for object to be detected
cv.createTrackbar("Area (min)", "Parameters", 5000, 100000, lambda _: ...)
# Maximum area for an object to be detected
cv.createTrackbar("Area (max)", "Parameters", 50000, 100000, lambda _: ...)

# Get background to eliminate
gray_median_frame: np.ndarray[np.uint8] = camera_one.get_median_frame()

while True:
    # Read each frame
    success: bool
    image_original: np.ndarray[np.uint8]
    success, image_original = camera_one.capture.read()

    image_HSV = cv.cvtColor(image_original, cv.BGR2HSV)

    h_min = cv.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv.getTrackbarPos("Val Min","TrackBars")
    v_max = cv.getTrackbarPos("Val Max","TrackBars")
    lower = np.array(h_min,s_min,v_min)
    upper = np.array(h_max,s_max,v_max)
    colormask = cv.inRange(image_HSV,lower,upper)
    image_result = cv.bitwise_and(image_original,image_original,mask=colormask)
    
    cv.imshow("HSV",image_HSV)
    cv.imshow("Mask",mask)
    cv.imshow("Masked_picture",image_result)


    # Check if stream is working properly
    if not success:
        logger.error("Can't receive frame (stream end?). Exiting ...")
        break

    # Blur the image to better detect edges
    image_blurred: np.ndarray[np.uint8] = cv.GaussianBlur(image_original, (7, 7), 1)

    # Convert image to grayscale to better detect edges
    image_grayscale: np.ndarray[np.uint8] = cv.cvtColor(
        image_blurred, cv.COLOR_BGR2GRAY)

    while(1): 
        # It converts the BGR color space of image to HSV color space 
        hsv = cv.cvtColor(image_blurred, cv.COLOR_BGR2HSV) 
      
        # Threshold of blue in HSV space 
        lower_blue = np.array([60, 35, 140]) 
        upper_blue = np.array([180, 255, 255]) 
  
        # preparing the mask to overlay 
        mask = cv.inRange(hsv, lower_blue, upper_blue) 
      
        # The black region in the mask has the value of 0, 
        # so when multiplied with original image removes all non-blue regions 
        result = cv.bitwise_and(image_blurred, image_blurred, mask = mask) 
  
        cv.imshow('frame', image_blurred) 
        cv.imshow('mask', mask) 
        cv.imshow('result', result) 
      
        cv.waitKey(0) 

    # Calculate absolute difference of current image and median frame
    image_differences: np.ndarray[np.uint8] = cv.absdiff(
        image_grayscale, gray_median_frame) 

    # Threshold to binarize
    _threshold: float
    image_differences: np.ndarray[np.uint8]
    _threshold, image_differences = cv.threshold(
        image_differences, 30, 255, cv.THRESH_BINARY)

    # Get threshold values from trackbar
    threshold1: int = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2: int = cv.getTrackbarPos("Threshold2", "Parameters")

    # Detect edges using Canny
    image_canny: np.ndarray[np.uint8] = cv.Canny(
        image_differences, threshold1, threshold2)

    # using a findContours() function 
    contours, _ = cv.findContours( 
        image_differences, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
                
    # Remove overlaps and noise using dilation
    kernel: np.ndarray[np.float64] = np.ones((5, 5))
    image_dilation: np.ndarray[np.uint8] = cv.dilate(image_canny, kernel, iterations=1)

    # Get contours of image to detect objects
    image_objects: np.ndarray[np.uint8] = image_original.copy()
    centroids: list[tuple[int, int]] = Camera.get_contours(
        image_dilation, image_objects)

    # Put text on each image to identify them
    cv.putText(image_original, "Original Image", (10, 25),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_differences, "Background removed", (10, 25),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_dilation, "Edge detection using Canny", (10, 25),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)
    cv.putText(image_objects, "Objects detected", (10, 25),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 200, 0), 2)

    # Stack the images into one window
    image_stack: np.ndarray[np.uint8]
    image_stack = stack_images(0.8, ([image_original, image_differences],
                                     [image_dilation, image_objects]))

    # Display each frame
    cv.imshow("Result", image_stack)

    if cv.waitKey(WAIT_KEYPRESS_MSEC) & MODIFIER_MASK == ESC_KEY:
        break

# Release the capture
camera_one.capture.release()
cv.destroyAllWindows()
