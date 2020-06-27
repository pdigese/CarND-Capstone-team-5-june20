import rospy
from tools import Model_tl
from styx_msgs.msg import TrafficLight
from keras.models import load_model
import tensorflow as tf
import numpy as np
import rospy
import cv2
import os


DIR = os.path.dirname(os.path.realpath(__file__))

class TLClassifierRL(object):
    def __init__(self):

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """