"""Module that contains the Robotic_Arm class."""

import time
import sys
import logging
import datetime
from Adafruit_PCA9685 import PCA9685
from robotarm.robot.hand import Hand
from robotarm.robot.joint import Joint

# Initialize the logger
logger = logging.getLogger("__arm__")
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

class RoboticArm:
    """Robotic Arm class."""

    def __init__(self):
        """Initialize the robotic arm."""
        self.pulse_width_modulation: PCA9685 = PCA9685()
        self.pulse_width_modulation.set_pwm_freq(60)
        self.hand: Hand = Hand(self.pulse_width_modulation, 0)
        self.base: Joint = Joint(self.pulse_width_modulation, 4, [100, 670], 410)
        self.shoulder: Joint = Joint(self.pulse_width_modulation, 3, [100, 650], 240)
        self.elbow: Joint = Joint(self.pulse_width_modulation, 2, [235, 680], 300)
        self.wrist: Joint = Joint(self.pulse_width_modulation, 1, [155, 555], 265)

        time.sleep(3)
        self.drop_object()
        time.sleep(1)

    def switch_off(self) -> None:
        """Determine the robotic arm."""
        self.open_hand()
        PCA9685()
        logger.info("The robotic arm has been determined.")

    def open_hand(self) -> None:
        """Open the hand servo."""
        self.hand.open()
        logger.info("Hand has been opened!")

    def close_hand(self) -> None:
        """Close the hand servo."""
        self.hand.close()
        logger.info("Hand has been closed!")

    def rotate(self, movement: int, left_rotation: bool = True) -> None:
        """
        Rotate the robotic arm.

        The first argument is the movement the base servo needs to perform.
        The second argument is an optional argument indicating in which direction
        the base should move. Standard is a rotation to the left.
        """
        self.base.move_to_position(self.base.position +
                                   (movement if left_rotation else -1 * movement))

    def move_shoulder(self, new_position: int) -> None:
        """
        Move the shoulder servo.

        The argument is the new position that the shoulder servo needs to move to.
        """
        self.shoulder.move_to_position(new_position)

    def move_elbow(self, new_position: int) -> None:
        """
        Move the elbow servo.

        The argument is the new position that the elbow servo needs to move to.
        """
        self.elbow.move_to_position(new_position)

    def move_wrist(self, new_position: int) -> None:
        """
        Move the wrist servo.

        The argument is the new position that the wrist servo needs to move to.
        """
        self.wrist.move_to_position(new_position)

    def drop_object(self) -> None:
        """Move the robotic arm to a drop place after picking up an object."""
        # Raise the arm
        self.move_shoulder(300)
        time.sleep(1)
        self.base.move_to_position(135)
        time.sleep(.25)
        self.move_wrist(225)
        self.move_elbow(300)
        time.sleep(.25)
        self.move_shoulder(240)
        time.sleep(.25)
        self.open_hand()
        logger.info("Drop zone has been reached!")

    def go_to_starting_position(self) -> None:
        """Move the robotic arm to its starting position (after dropping an object)."""
        self.move_shoulder(300)
        time.sleep(1)
        self.base.move_to_position(425)
        time.sleep(.5)
        self.move_elbow(320)
        time.sleep(1)
        self.move_shoulder(250)
        time.sleep(1)
        self.move_wrist(315)
        time.sleep(.5)
        logger.info("Starting position has been reached!")

    def move(self, distance: float, forward: bool = True) -> None:
        """
        Move the robotic arm forward or backwards.

        The first parameter is the distance that robotic arm needs
        to move (in centimeters).
        The second parameter is the direction. Standard is forward.
        """
        normalized_distance: float = distance / .6
        direction: int = 1 if forward else -1

        if 0 < distance < 3:
            rotation = [(servo_movement)
                for servo_movement in (-35, 0, 10)] 
        elif 3 < distance < 5.5:
            rotation = [(servo_movement)
                for servo_movement in (0, 0, 20)]
        else:
            rotation = [int(direction * normalized_distance * servo_movement)
                for servo_movement in (6, -2.5, 5.6)] 
        
        # Move the respective servos
        if forward:
            for x in range(10):
                self.move_elbow(self.elbow.position + rotation[0]/10)
                time.sleep(.1)
            for x in range(10):
                self.move_shoulder(self.shoulder.position + rotation[1]/10)
                time.sleep(.1)
            for x in range(10):
                self.move_wrist(self.wrist.position + rotation[2]/10)
                time.sleep(.1)
        else:
            self.move_shoulder(300)    
            self.move_wrist(225)
            self.move_elbow(300)

    def rotate_by_degree(self, degree: float) -> None:
        """
        Rotate the robotic arm by a given degree.

        The argument is the rotation degree. (Positive values result in a rotation
        to the left, negative values result in a rotation to the right.)
        """
        rotation: int = int(17 / 6 * degree)
        self.base.move_to_position(self.base.position + rotation)

    def get_all_positions(self) -> None:
        """Print the current positions of all the servos."""
        print("\nCurrent position of all the servos",
              "\n\tBase:\t\t", self.base.position,
              "\n\tShoulder:\t", self.shoulder.position,
              "\n\tElbow:\t\t", self.elbow.position,
              "\n\tWrist:\t\t", self.wrist.position)
