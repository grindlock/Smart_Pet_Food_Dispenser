# Pet Food Dispenser 

This project is for the COP 5611 - Operating Systems Design Principles course. The project motivation comes for trying to use object detection in conjuction with I/O devices, in this case a PIR motion sensor, a camera and a stepper motor. The objection detection is processed on the device thanks to TensoFlow Lite and uses the Coco v1 model. I used the tutorial and modify the example code provided by Digi-Key. Here is the link to the tutorial [How to Perform Object Detection with TensorFlow Lite on Raspberry Pi](https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588).

**The following files are included:**
* pet_food_dispenser.py 
* coco_ssd_mobilenet_v1_edited.zip
* Cat Food Dispenser.pptx
* Demo_Video.mp4

**Parts List:**
* Raspberry pi 
* PIR motion sensor (prefebly the one that comes in a breakout board)
* PiCamera 2 
* NEMA 17 stepper motor with at least 1A and 12v
* DVR228825 stepper motor driver
* Power source for the motor 12V and between 1A and 2A depending of the motor.

**Installation:**

Install Python 3.8 and pip3
Install the following libraries:
libjpeg-dev 
libtiff5-dev 
libjasper-dev 
libpng12-dev 
libavcodec-dev 
libavformat-dev 
libswscale-dev 
libv4l-dev 
libxvidcore-dev 
libx264-dev
qt4-dev-tools 
libatlas-base-dev 
libhdf5-103

Using pip install OpenCV
opencv-contrib-python==4.1.0.25

To install TensorFlow Lite (https://www.tensorflow.org/lite/guide/python):
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install python3-tflite-runtime

**How to use the program:**

first unzip coco_ssd_mobilenet_v1_edited.zip in the same directory where the program is. Name the directory coco_ssd_mobilenet_v1. If you change the name of the directory you need to change the varible that contains the name of the directory in the program.

In terminal type $ python3 pet_food_dispenser.py

The program start opening an OpenCV window to show the camera and the terminal will print messages for the different process.
