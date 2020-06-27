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

class TLClassifierSim(object):
    def __init__(self):
        #TODO load classifier

        weight = 'weights.epoch-06__loss-0.02387__.hdf5'
        weight_dir = 'model'
        self.model = Model_tl()

        self.model.load_weights(os.path.join(DIR,weight_dir,weight))
        self.graph = tf.get_default_graph()
        self.state = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        tl_dict = {0:TrafficLight.RED, 1:TrafficLight.YELLOW, 2:TrafficLight.GREEN, 3:TrafficLight.UNKNOWN }

        image = cv2.resize(image,(100,100))
        image = np.array([image])
        with self.graph.as_default():
            output = (self.model.predict(image)).squeeze()
            y_pred = np.argmax(output)
            conf = output[y_pred]
            rospy.logerr('confidence:%s',conf)
            if conf > 0.5:
                self.state = tl_dict[y_pred]
                return self.state

            return TrafficLight.UNKNOWN
