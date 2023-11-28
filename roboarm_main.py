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
from robotarm.camera.calculate_movement import calculate_rotation_degrees
from robotarm.camera.calculate_movement import calculate_distance
from robotarm.camera.stack_images import stack_images

# Import the robotic arm and the camera class
from robotarm.robot.robotic_arm import RoboticArm
from robotarm.camera.camera import Camera
from robotarm.robot.robotic_arm_adjust import check_inside
from robotarm.robot.robotic_arm_adjust import rotate_points
from robotarm.robot.robotic_arm_adjust import calculate_adjust_angle

# Initialize the logger
logger = logging.getLogger(__name__)
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

# Initialize the robotic arm
arm: RoboticArm = RoboticArm()

# Initialise the camera
cam: Camera = Camera(640, 480, cv.VideoCapture(0))

# Initialize constants to exit the program
WAIT_KEYPRESS_MSEC: int = 20
MODIFIER_MASK: int = 0xFF
ESC_KEY: int = 27

# Check if the capturing works
if not cam.capture.isOpened():
    logger.critical("Capture not opened! Exiting...")
    arm.switch_off()
    logger.info("Determined the robotic arm successfully!")

    cam.capture.release()
    cv.destroyAllWindows()
    logger.info("Determined the camera successfully!")
    sys.exit()

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

# Get background to eliminate
gray_median_frame: np.ndarray[np.uint8] = cam.get_median_frame()

# Variable to determine if it is the first loop
FIRST_LOOP: bool = True

# Variable to determine if objects and when they are found
OBJECTS_FOUND_AT: int = 0
OBJECTS_FOUND: bool = False

# Default/Starting position of the hand
HAND_POSITION: tuple[int, int] = (346, 236)
BASE_POSITION: tuple[int, int] = (311, 2)
HAND_POLY: list[tuple[float,float]] = [(314,252),(339,251),(342,449),(318,450)]
DISTANCE_BASE_TO_HAND: int = int(((HAND_POSITION[0] - BASE_POSITION[0]) ** 2 +
                                  (HAND_POSITION[1] - BASE_POSITION[1]) ** 2) ** .5)

while True:
    # Give the camera a frame to correctly initialize
    if FIRST_LOOP:
        FIRST_LOOP = False
        continue

    # Read each frame
    success: bool
    image_original: np.ndarray[np.uint8]
    success, image_original = cam.capture.read()

    # Check if the stream is working properly
    if not success:
        arm.switch_off()
        logger.info("Dertermined the robotic arm successfully!")

        cam.capture.release()
        cv.destroyAllWindows()
        logger.info("Determined the camera successfully!")

        sys.exit()

    # Blur the image to better detect edges
    image_blurred: np.ndarray[np.uint8] = cv.GaussianBlur(image_original, (7, 7), 1)

    # Convert image to grayscale to better detect edges
    image_grayscale: np.ndarray[np.uint8] = cv.cvtColor(
        image_blurred, cv.COLOR_BGR2GRAY)

    # Calculate absolute difference of current image and median frame
    image_difference: np.ndarray[np.uint8] = cv.absdiff(
        image_grayscale, gray_median_frame)

    # Creating a mask to stop detection from outside the area
    mask=np.zeros(image_difference.shape[:2], dtype="uint8")
    cv.circle(mask,(325,7),434,255,-1)
    image_masked: np.ndarray[np.uint8]=cv.bitwise_and(
        image_difference, image_difference,mask=mask)

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

    # Check if objects are found (Detected for the first time)
    if len(centroids) > 0 and not OBJECTS_FOUND:
        OBJECTS_FOUND = True
        OBJECTS_FOUND_AT = time.time()

    # Check if objects are still found and enough time has passed
    if len(centroids) > 0 and OBJECTS_FOUND and time.time() - OBJECTS_FOUND_AT >= 5:
        logger.info("Heads to the point %s!", centroids[0])
        # Move the robotic arm to the center
        arm.go_to_starting_position()
        # Calculate the distances from the robotic arm to the objects
        distances: list[float] = calculate_distance(centroids, BASE_POSITION)
        # Calculate the rotation degrees the robotic arm needs to perform
        rotation_degrees: list[float] = calculate_rotation_degrees(
            centroids, distances, BASE_POSITION)
        # Calculate the distances from the hand to the objects
        distances_hand_to_objects: list[float] = [distance - DISTANCE_BASE_TO_HAND
                                                  for distance in distances]
        distances_hand_to_objects, centroids, rotation_degrees = sort_by_distance(distances_hand_to_objects,centroids,rotation_degrees)
        if distances_hand_to_objects[0] < 0 or distances_hand_to_objects[0] >= 190:
            logger.error("Unreachable distance!")
        # Rotate the arm to the object
        logger.info("Rotates %s degrees!", rotation_degrees[0])
        arm.rotate_by_degree(rotation_degrees[0])
        time.sleep(1)
        new_hand_poly: list[tuple[float, float]]
        adjust_angle: float
        new_hand_poly = rotate_points(HAND_POLY, -rotation_degrees[0], BASE_POSITION)
        if not check_inside(new_hand_poly, len(new_hand_poly), centroids[0]):
            logger.info("Centroid is outside the polygon! Adjust...")
            adjust_angle = calculate_adjust_angle(new_hand_poly, centroids[0], BASE_POSITION)
            logger.info("Adjusted by %s degree", -adjust_angle)
            arm.rotate_by_degree(-adjust_angle)
            time.sleep(1)
        else:
            logger.info("Centroid lies within the polygon!")

        # Move the robotic arm forwards
        arm.move(distances_hand_to_objects[0] / 10)
        time.sleep(1)
        # Grab the object
        arm.close_hand()
        time.sleep(1)
        # Move the robotic arm backwards
        arm.move(distances_hand_to_objects[0] / 10, forward=False)
        # Bring the object to the dropping place and drop it
        arm.drop_object()
        time.sleep(1)
        OBJECTS_FOUND = False
        OBJECTS_FOUND_AT = 0
        FIRST_LOOP = True

    # Determine the script by pressing the ESC-key
    if cv.waitKey(WAIT_KEYPRESS_MSEC) & MODIFIER_MASK == ESC_KEY:
        arm.go_to_starting_position()
        break

# Exit the program
logger.info("Program ended. Exiting...")
arm.switch_off()
logger.info("Determined the robotic arm successfully!")
cam.capture.release()
cv.destroyAllWindows()
logger.info("Determined the camera successfully!")