from styx_msgs.msg import TrafficLight
from sensor_msgs.msg import Image
import tensorflow as tf
import numpy as np
import os
import rospy
from cv_bridge import CvBridge
from PIL import ImageDraw
from skimage.color import rgb2hsv

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

        # TODO: use self.classify_light_color

        if scores[0] > self.threshold:
            if classes[0] == 1:
                #rospy.logwarn("GREEN with confidency : {}".format(scores[0]))
                return TrafficLight.GREEN
            elif classes[0] == 2:
                #rospy.logwarn("RED with confidency : {}".format(scores[0]))
                return TrafficLight.RED
            elif classes[0] == 3:
                #rospy.logwarn("YELLOW with confidency : {}".format(scores[0]))
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
                #rospy.logwarn("RED identified")
                class_ = 2
            if (middle_sub_img_brightness > lower_sub_img_brightness) and (middle_sub_img_brightness > upper_sub_img_brightness):
                #rospy.logwarn("YELLOW identified")
                class_ = 3
            if (upper_sub_img_brightness > middle_sub_img_brightness) and (upper_sub_img_brightness > lower_sub_img_brightness):
                #rospy.logwarn("GREEN identified")
                class_ = 1
        
            #self.sub_img_debug_helper(sub_img)

        return class_

    """ The solution below was found on https://www.kaggle.com/photunix/classify-traffic-lights-with-pre-trained-cnn-model """
    def classify_light_color(self, rgb_image):
        """
            Full pipeline of classifying the traffic light color from the traffic light image

            :param rgb_image: the RGB image array (height,width, RGB channel)
            :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
            """

        hue_1d_deg = self.get_masked_hue_values(rgb_image)

        if len(hue_1d_deg) == 0:
            return 0

        hue_1d_rad = self.convert_to_hue_angle(hue_1d_deg)

        return self.classify_color_by_range(hue_1d_rad)

    def classify_color_by_range(self, hue_value):
        """
        Determine the color (red, yellow or green) in a hue value array

        :param hue_value: hue_value is radians
        :return: the color index ['red', 'yellow', 'green', '_', 'unknown']
        """

        red_index, green_index, yellow_index = self.get_rgy_color_mask(hue_value)

        color_counts = np.array([np.sum(red_index) / len(hue_value),
                                 np.sum(yellow_index) / len(hue_value),
                                 np.sum(green_index) / len(hue_value)])

        # TODO: this could use a nicer approach
        color_text = [2, 3, 1]

        min_index = np.argmax(color_counts)

        return color_text[min_index]

    def get_rgy_color_mask(self, n_hue_value):
        """
        return a tuple of np.ndarray that sets the pixels with red, green and yellow matrices to be true

        :param hue_value:
        :return:
        """

        red_index = np.logical_and(n_hue_value < (0.125 * np.pi), n_hue_value > (-0.125 * np.pi))

        green_index = np.logical_and(n_hue_value > (0.66 * np.pi), n_hue_value < np.pi)

        yellow_index = np.logical_and(n_hue_value > (0.25 * np.pi), n_hue_value < (5.0 / 12.0 * np.pi))

        return red_index, green_index, yellow_index

    def convert_to_hue_angle(self, hue_array):
        """
        Convert the hue values from [0,179] to radian degrees [-pi, pi]

        :param hue_array: array-like, the hue values in degree [0,179]
        :return: the angles of hue values in radians [-pi, pi]
        """

        hue_cos = np.cos(hue_array * np.pi / 90)
        hue_sine = np.sin(hue_array * np.pi / 90)

        hue_angle = np.arctan2(hue_sine, hue_cos)

        return hue_angle

    def get_masked_hue_values(self, rgb_image):
        """
        Get the pixels in the RGB image that has high saturation (S) and value (V) in HSV chanels

        :param rgb_image: image (height, width, channel)
        :return: a 1-d array
        """

        hsv_test_image = rgb2hsv(rgb_image)
        s_thres_val = self.channel_percentile(hsv_test_image[:, :, 1], percentile=30)
        v_thres_val = self.channel_percentile(hsv_test_image[:, :, 2], percentile=70)
        val_mask = self.high_value_region_mask(hsv_test_image, v_thres=v_thres_val)
        sat_mask = self.high_saturation_region_mask(hsv_test_image, s_thres=s_thres_val)
        masked_hue_image = hsv_test_image[:, :, 0] * 180
        # Note that the following statement is not equivalent to
        # masked_hue_1d= (maksed_hue_image*np.logical_and(val_mask,sat_mask)).ravel()
        # Because zero in hue channel means red, we cannot just set unused pixels to zero.
        masked_hue_1d = masked_hue_image[np.logical_and(val_mask, sat_mask)].ravel()

        return masked_hue_1d

    def channel_percentile(self, single_chan_image, percentile):
        sq_image = np.squeeze(single_chan_image)
        assert len(sq_image.shape) < 3

        thres_value = np.percentile(sq_image.ravel(), percentile)

        return float(thres_value) / 255.0

    def high_saturation_region_mask(self, hsv_image, s_thres=0.6):
        if hsv_image.dtype == np.int:
            idx = (hsv_image[:, :, 1].astype(np.float) / 255.0) < s_thres
        else:
            idx = (hsv_image[:, :, 1].astype(np.float)) < s_thres
        mask = np.ones_like(hsv_image[:, :, 1])
        mask[idx] = 0
        return mask