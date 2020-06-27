import rospy
from tools import Model_tl
from styx_msgs.msg import TrafficLight
from keras.models import load_model
import tensorflow as tf
import numpy as np
import rospy
import cv2
import os

SSD_GRAPH_FILE = "/../../../nn/frozen_inference_graph.pb"

class TLClassifierRL(object):
    def __init__(self):
		self.graph = self.load_graph(os.path.dirname(os.path.abspath(__file__)) + SSD_GRAPH_FILE)

		self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
		with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            (detection_boxes, detection_scores, detection_classes) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: img_expand})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN

	def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
				
				self.image_tensor = graph.get_tensor_by_name("image_tensor:0")
				self.detection_boxes = graph.get_tensor_by_name(
					"detection_boxes:0"
				)
				self.detection_scores = graph.get_tensor_by_name(
					"detection_scores:0"
				)
				self.detection_classes = graph.get_tensor_by_name(
					"detection_classes:0"
				)
        return graph