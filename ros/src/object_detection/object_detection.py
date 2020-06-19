#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose
import tensorflow as tf
import os
from PIL import (
    Image as Img,
)  # Must not be named image since there is a datatype called Image from ROS
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from cv_bridge import CvBridge
import json
from styx_msgs.msg import TrafficLightArray, TrafficLight
import yaml


SSD_GRAPH_FILE = "/../../../nn/frozen_inference_graph.pb"

"""
TODO: Description goes here.
"""

"""
Note: All functions (whih are not belonging to the class itself) are copied from:
- load_graph, filter_boxes and to_image_coords:
https://github.com/udacity/CarND-Object-Detection-Lab/blob/master/CarND-Object-Detection-Lab.ipynb
- pipeline:
https://github.com/udacity/CarND-Object-Detection-Lab/blob/e91d0de6cd54834966cdb06b0172e23f8a0c124f/exercise-solutions/e5.py
"""

# FIXME: needs to go somewhere in an encapsulation
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])


class ObjDetection(object):
    def __init__(self):
        rospy.init_node("object_detection")

        self.latest_img = None
        self.cnt = 0
        self.latest_tls = None  # latest traffic light state
        self.curr_pose = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # subscribe the camera topic
        # publish what? traffic light? other objects?
        # load coco
        self.detection_graph = self.load_graph(
            os.path.dirname(os.path.abspath(__file__)) + SSD_GRAPH_FILE
        )
        annotation_path = os.path.dirname(os.path.abspath(__file__)) + SSD_GRAPH_FILE

        self.sub_img = rospy.Subscriber("/image_raw", Image, self.image_cb)
        self.sub_tls = rospy.Subscriber(
            "/vehicle/traffic_lights", TrafficLightArray, self.tls_cb
        )
        self.sub_pos = rospy.Subscriber("/current_pose", PoseStamped, self.pose_cb)

        self.pub_detected = rospy.Publisher("/image_detection", Image, queue_size=1)
        self.pub_cropped_img = rospy.Publisher("/image_cropped", Image, queue_size=10)

        rospy.loginfo("Coco has been initialized successfully")
        self.loop()

    def loop(self):
        rate = rospy.Rate(
            2
        )  # 1Hz TODO: What's a goo frequency here? It's an heavy job...
        with tf.Session(graph=self.detection_graph) as self.sess:
            self.image_tensor = self.sess.graph.get_tensor_by_name("image_tensor:0")
            self.detection_boxes = self.sess.graph.get_tensor_by_name(
                "detection_boxes:0"
            )
            self.detection_scores = self.sess.graph.get_tensor_by_name(
                "detection_scores:0"
            )
            self.detection_classes = self.sess.graph.get_tensor_by_name(
                "detection_classes:0"
            )
            while not rospy.is_shutdown():

                if self.latest_img is not None:
                    processed_image = self.pipeline(self.latest_img)
                    pub_img = Image()
                    bridge = CvBridge()
                    """
                    rqt_image_view reports with pathtrough:
                    ImageView.callback_image() could not convert image from '8UC3' to 'rgb8' ([8UC3] is not
                    a color format. but [rgb8] is. The conversion does not make sense).
                    => therefore encoding
                    """
                    image_message = bridge.cv2_to_imgmsg(
                        processed_image, encoding="rgb8"
                    )  # required by the rqt_image_viewer
                    self.pub_detected.publish(image_message)
                rate.sleep()

    def pose_cb(self, msg):
        self.curr_pose = msg

    def image_cb(self, msg):
        bridge = CvBridge()
        # for conversion please read:
        # http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        self.latest_img = bridge.imgmsg_to_cv2(
            msg, desired_encoding="rgb8"
        )  # TODO: coco requires rgb as far as I know

    def tls_cb(self, msg):
        # receives a whole list of traffic lights
        self.latest_tls = msg.lights

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = COLOR_LIST[class_id]
            class_label = "NONE"
            if classes[i] == 1:
                class_label = "GREEN"
            elif classes[i] == 2:
                class_label = "RED"
            elif classes[i] == 3:
                class_label = "YELLOW"
            draw.line(
                [(left, top), (left, bot), (right, bot), (right, top), (left, top)],
                width=thickness,
                fill=color,
            )
            draw.text((left, top), "{} Score: {}".format(class_label, scores[i]))

    def get_closest_tl_state(self, pose):
        """
        Returns the state of the closest traffic light
        """
        # contains the stop_line_positions off all traffic lights
        min_dist = 9999999
        tlc_idx = 0
        stop_line_pos = self.config["stop_line_positions"]
        for i, tl_pos in enumerate(stop_line_pos):
            # just check the eucledian distance to other traffic lights
            distance = np.sqrt(
                (tl_pos[0] + pose.pose.position.x) + (tl_pos[1] + pose.pose.position.y)
            )
            if min_dist > distance:
                min_dist = distance
                tlc_idx = i
        return self.latest_tls[i].state

    def pipeline(self, img):
        """
        FIXME: I am just copied, but to make a proper job, I need to return either
        - just the region of interest (so the traffic light itself)
        - OR the information about the status of the traffic light.
        """
        draw_img = Img.fromarray(img)
        boxes, scores, classes = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: np.expand_dims(img, 0)},
        )
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.2

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(
            confidence_cutoff, boxes, scores, classes
        )

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = draw_img.size
        box_coords = self.to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        self.draw_boxes(draw_img, box_coords, classes, scores)
        return np.array(draw_img)


if __name__ == "__main__":
    try:
        ObjDetection()
    except rospy.ROSInterruptException:
        rospy.logerr("Could not start object detection node.")
