# Smart Pet Food Dispenser (configure to use with a cat)
# Author: Sergio Perez-Aponte
# Date: 30th April 2021
#
# This code uses PIR motion sensor to trigger objection detection if the object in
# question is in focus then the stepper motor runs. After the script sleeps for 10 minutes.
# TensorFlow and OpenCV are use to detect for the pet and shoe the camera. Some of the
# object detection code comes from exaples by Evan Juras (EdgeElectronics),
# Adrian Rosebrock (PyImageSearch) and the TensorFlow repo.
#

# Import packages
import os
import cv2
import numpy as np
import sys
from time import sleep
from threading import Thread
import importlib.util
from gpiozero import MotionSensor, OutputDevice as step_motor

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

MODEL_NAME = 'coco_ssd_mobilenet_v1'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
imW, imH = 640, 480
pir = MotionSensor(4)
# Need to make the motor run on its on thread for the sleep
in1 = step_motor(17)
in2 = step_motor(27)
in3 = step_motor(22)
# 1/4 of the step
in1.off()
in2.on()
in3.off()
direction_pin = step_motor(25)
step_pin = step_motor(8)
# Clockwise rotation
direction_pin.on()
# necesary sleep for the motor between steps
step_delay = (0.005/4)
# motor takes 200 steps for one revolution.
# To dispense one cup of food it takes 6 revolutions
# To dispense more than 1 cup multiply by the numbers
# of cups. Likewise for less than one cup divide by the number
one_cup = (200 * 6)
# The number of steps the motor must make
num_steps = (one_cup)
# Sleep for 10 minutes
motor_sleep = (60 * 10)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime,
# else import from regular tensorflow

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
sleep(1)

# Function to run a bipolar stepper motor
def run_motor(step_delay, num_steps, motor_sleep):

    for x in range(num_steps):
        step_pin.on()
        sleep(step_delay)
        step_pin.off()
        sleep(step_delay)
    print("Program goes to sleep for 10 minutes")
    sleep(motor_sleep)
    print("Program resumes.")

while True:

    # Timer to calculate the frame rate
    t1 = cv2.getTickCount()

    one_frame = videostream.read()

    # Process the frame
    frame = one_frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # Display the process frame
    cv2.imshow('Pet Dispenser Output', frame)

    if pir.motion_detected:
        print("Motion Detected. Let's see if it is a cat.")

        # Tensorflow lite perform the detection
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve result from tensorflow lite
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects

        # Let see if the result is a cat or for debbug a teddy bear
        if (labels[int(classes[0])] == "cat" or labels[int(classes[0])]== "teddy bear"):
            print("The resul is: " + labels[int(classes[0])])

            run_motor(step_delay, num_steps, motor_sleep)

    # Calculate the framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Terminate the video thread and close the cv window
cv2.destroyAllWindows()
videostream.stop()