from styx_msgs.msg import TrafficLight
from sensor_msgs.msg import Image
import tensorflow as tf
import numpy as np
import os
import rospy
from cv_bridge import CvBridge
from PIL import ImageDraw

from PIL import (
    Image as Img,
)  # Must not be named image since there is a datatype called Image from ROS


SSD_GRAPH_FILE = "/../../../../nn/frozen_inference_graph.pb"


class TLClassifierRL(object):
    def __init__(self):
        self.graph = self.load_graph(os.path.dirname(os.path.abspath(__file__)) + SSD_GRAPH_FILE)
        self.sess = tf.Session(graph=self.graph)
        self.threshold = 0.6
        self.debug_output_stream = rospy.Publisher('/image_color_debug', Image, queue_size=1)

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

        scores = np.squeeze(detection_scores)
        boxes = np.squeeze(detection_boxes)
        classes = np.squeeze(detection_classes).astype(np.int32)

        classes[0] = self.pix_brightness_cntr(image, scores[0], boxes[0])

        self.classifier_debug_helper(image, boxes[0], classes[0], scores[0])

        if scores[0] > self.threshold:
            if classes[0] == 1:
                rospy.logwarn("GREEN with confidency : {}".format(scores[0]))
                return TrafficLight.GREEN
            elif classes[0] == 2:
                rospy.logwarn("RED with confidency : {}".format(scores[0]))
                return TrafficLight.RED
            elif classes[0] == 3:
                rospy.logwarn("YELLOW with confidency : {}".format(scores[0]))
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

    def classifier_debug_helper(self, image, box, cl, score):
        """ adds the identified box to the image """

        bridge = CvBridge()
        """
        rqt_image_view reports with pathtrough:
        ImageView.callback_image() could not convert image from '8UC3' to 'rgb8' ([8UC3] is not
        a color format. but [rgb8] is. The conversion does not make sense).
        => therefore encoding
        """

        image = self.draw_box(image, box, cl, score, thickness=4)

        image_message = bridge.cv2_to_imgmsg(
            image, encoding="rgb8"
        )
        self.debug_output_stream.publish(image_message)

    def draw_box(self, image, box, class_, score, thickness=4):
        """Draw bounding boxes on the image"""
        draw_img = Img.fromarray(image)
        draw = ImageDraw.Draw(draw_img)

        width, height = draw_img.size
        bot = box[0] * height
        left = box[1] * width
        top = box[2] * height
        right = box[3] * width

        if score > self.threshold:
            color = (int(class_)//2*255, int(class_)%2*255, 0)
            draw.line(
                [(left, top), (left, bot), (right, bot), (right, top), (left, top)],
                width=thickness,
                fill=color,
            )
            draw.text((left, top), "{}".format(round(1E4*score)/100))
        return np.array(draw_img)


    def sub_img_debug_helper(self, sub_img):
        bridge = CvBridge()
        image_message = bridge.cv2_to_imgmsg(
            sub_img, encoding="rgb8"
        )
        self.debug_output_stream.publish(image_message)

    def pix_brightness_cntr(self, image, score, box):
        """ alternative light state identifier """
        class_ = 0

        if score > self.threshold:
            draw_img = Img.fromarray(image)
            width, height = draw_img.size
            bot = int(box[0] * height)
            left = int(box[1] * width)
            top = int(box[2] * height)
            right = int(box[3] * width)


            sub_img = image[bot:top, left:right, :]
            lower_third = int((top - bot) * 1./3.)
            upper_third = int((top - bot) * 2./3.)

            lower_sub_img_brightness = np.sum(sub_img[:lower_third, :, :])
            middle_sub_img_brightness = np.sum(sub_img[lower_third:upper_third, :, :])
            upper_sub_img_brightness = np.sum(sub_img[upper_third:, :, :])

            if (lower_sub_img_brightness > middle_sub_img_brightness) and (lower_sub_img_brightness > upper_sub_img_brightness):
                rospy.logwarn("RED identified")
                class_ = 2
            if (middle_sub_img_brightness > lower_sub_img_brightness) and (middle_sub_img_brightness > upper_sub_img_brightness):
                rospy.logwarn("YELLOW identified")
                class_ = 3
            if (upper_sub_img_brightness > middle_sub_img_brightness) and (upper_sub_img_brightness > lower_sub_img_brightness):
                rospy.logwarn("GREEN identified")
                class_ = 1
        
            #self.sub_img_debug_helper(sub_img)

        return class_
            