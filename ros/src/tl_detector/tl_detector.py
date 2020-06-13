#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 2
LOOK_AHEAD = 200

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb,queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb,queue_size=1)

        # List of positions that correspond to the line to stop in front of for a given intersection
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_wp_list = None

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints
        self.stop_wp_list = self.get_stop_line_wp(self.stop_line_positions)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        #if self.state != state:
        #    self.state_count = 0
        #    self.state = state
        #    light_wp = -1
        #elif self.state_count >= STATE_COUNT_THRESHOLD:
        #    self.last_state = self.state
        #    light_wp = light_wp if state == TrafficLight.RED else -1
        #    self.last_wp = light_wp
        #else:
        #    light_wp = -1

        if state != TrafficLight.RED:
            light_wp = -1
        
        # note: The video says to publish the stopline waypoint, however, the written instruction
        # tells us to publish the index of the traffic light waypoint (ranging from 0...NUM_OF_TRAFFIC_LIGHTS-1)
        # However, the planner is not interested in the index of the traffic line but in the index of the stop line
        # where it is supposed to stop.
        # gerr('published stopline idx: %d',light_wp)
        if state == TrafficLight.RED:
            rospy.logerr('closest_light_index:%d, RED',light_wp)
        if state == TrafficLight.YELLOW:
            rospy.logerr('closest_light_index:%d, YELLOW',light_wp)
        if state == TrafficLight.GREEN:
            rospy.logerr('closest_light_index:%d, GREEN',light_wp)
        self.upcoming_red_light_pub.publish(Int32(light_wp))
        self.state_count += 1

    def get_closest_waypoint(self, position):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

        min_dist = float('inf')
        index = None
        #    dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        pos_x = position[0]
        pos_y = position[1]

        dl = lambda x1,y1,x2,y2 : math.sqrt((x1-x2)**2 + (y1-y2)**2 )

        for i in range(len(self.waypoints)-1):

            dist = min(min_dist, dl(pos_x,pos_y,self.waypoints[i].pose.pose.position.x,self.waypoints[i].pose.pose.position.y))
            
            if dist < min_dist:
                min_dist = dist
                index = i

        return index



    def get_light_state(self, idx=None):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN   ## originally False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
    
        #Get classification
        if idx == None:
            return self.light_classifier.get_classification(cv_image)
        else:
            # TODO: Implementation which takes the current state if the indexed traffic light
            return self.lights[idx].state

    def get_stop_line_wp(self,stop_line_positions):
    
        stop_line_index_list =[]
            
        for i in range(len(stop_line_positions)-1):
            stop_line_index = self.get_closest_waypoint(stop_line_positions[i])
            stop_line_index_list.append(stop_line_index)

        return stop_line_index_list


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
#        light = None


        #TODO find the closest visible traffic light (if one exists)

#        if light:
#            state = self.get_light_state(light)
#            return light_wp, state

        if self.pose and self.waypoints and self.stop_wp_list:
    #        rospy.logerr('loop active')
            car_x = self.pose.pose.position.x
            car_y = self.pose.pose.position.y
            car_wp = self.get_closest_waypoint([car_x,car_y])        
            
            min_delta_index = float('inf')
            closest_light_index = None
            tl_idx = None
    
            for i in range(len(self.stop_wp_list)-1):
                delta_index = self.stop_wp_list[i]-car_wp
                    
                if delta_index < 0:
                    delta_index += len(self.waypoints)     
                
                if delta_index < LOOK_AHEAD and delta_index < min_delta_index:
                    min_delta_index = delta_index
                    closest_light_index = self.stop_wp_list[i]
                    tl_idx = i
                    #rospy.logerr('closest_light_index:%d',closest_light_index)
            
            if closest_light_index:
                #rospy.logerr('closest_light_index:%d',closest_light_index)
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #        cv2.imwrite(("images/{}.jpg".format(float(rospy.get_time()))),cv_image)
                
                #state = self.get_light_state()
                state = self.get_light_state(tl_idx)   # FIXME: remove the argument to use camera tl detection!!!
                #rospy.logerr('light index:%d  state:%d',closest_light_index, state)
            
                return closest_light_index, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
