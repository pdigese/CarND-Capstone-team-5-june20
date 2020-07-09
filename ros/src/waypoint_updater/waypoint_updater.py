#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
from std_msgs.msg import Int32
#from scipy.interpolate import spline
from scipy import interpolate

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS_FOR_DECELERATION = 50
MAX_SPEED = 11.1111 # ms
STOPLINE_BUFFER = 2. # m
REMAINING_DIST_TOO_CLOSE_FOR_DECEL = 10.    # If the car is at full speed and less then this distance ahead of the stop line


class WaypointUpdater(object):
    '''
    Class for creating waypoints with the required longitudinal velocity information.
    Waypoints are already given by the '/base_waypoints' topic, where each waypoint contains
    a predefined speed of 11.1111 m/s. This speed needs to be set up based on the upcoming traffic
    light state so that the car stops at the traffic light stop lane.

    Once a traffic light is being reported, and the distance to the stop line is less than 
    LOOKAHEAD_WPS_FOR_DECELERATION waypoints, then the trajectory is planned such the car stops
    with its center 2 meters in front of the stop line.

    The deceleration is planned as follows:

    VELOCITY
    ^
    |        <--- MAX_VELOCITY --->              XXXXXXXXX|
    |                                   XXXXXXXXX
    |                             XXXXXX                  |
    |                       XXXXXX
    |                     XX                              |
    |                   XX
    |                  X                                  |
    |                XX
    |               X                                 +-- |
    |              X                          START OF
    |             X                           DECELERATION|
    |            X
    |           X                                         |
    |         XX
    |      XXX                                            |
    |  XXXX
    +XX---------------------------------------------------------->
    0                                         DISTANCE TO STOP LINE

    Note: Thes x-axis is the distance to the stop line AND not the time,
    therefore the actual velocity behavior over time looks different!

    The curve is defined by splines, and the max velocity and distance
    to stop line is normalized to 1.

    For each trajectory point this model is being used.
    Trjectoy points right of "START OF DECELERATION" keep their velocity
    (max velocity), whereas trajectory points left of this model are kept
    at zero velocity. Any other trajectory point's velocity is planned
    according to the model above.

    '''
    def __init__(self):
        '''
        Constructor for the Waypoint updater. Subscribes all required packages and
        start publishing the final waypoints, which need to be followed.
        '''
        self.waypoint_tree = None
        self.pose = None
        self.latest_stop_line_idx = -1
        self.decelleration_start_pos_already_found = False
        self.total_decel_len = 0
        self.curr_vel = None
        self.actual_accel_wp_end = None

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.car_curr_vel_cb)

        self.sub_tl_idx = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_light_idx_cb)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        
        # spline for the velocity deceleration
        x = np.array([-0.1, 0., .1, .4, 1., 1.1])
        y = np.array([0., 0., .1, .5, 1., 1.])
        self.vel_spline = interpolate.splrep(x, y, s=0)

        self.cyclic_traj_gen_and_publishing()

    def car_curr_vel_cb(self, msg):
        self.curr_vel = msg.twist.linear.x

    def traffic_light_idx_cb(self, msg):
        self.latest_stop_line_idx = msg.data

    def cyclic_traj_gen_and_publishing(self):
        '''
        Does cyclic trajectory generation based on the current pose and the base trajectory
        and publishes the planned trajectory.
        f_cyclic = 30 Hz since more seems not to be required, since the provided information is
        not updated more frequently.
        '''
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # ENDLESS LOOP
            if self.waypoint_tree and self.pose:
                wp_idx = self.get_closest_waypoint_idx() # gets the waypoint where we currently are
                # Publish the coming 50 trajectory points based on the vehicles current waypoint
                # ahead (wp_idx)
                self.publish_trajectory_v2(wp_idx)
            # "END" OF ENDLESS LOOP, wait the remaining time
            rate.sleep()
        pass

    def get_closest_waypoint_idx(self):
        '''
        Get the waypoint index which is
        - closest to the current waypoint (by KDTree query)
        - in front of the car (by checking if the pointing end of the vector, which is being made from the 
        current waypoint and the waypoint behind, is in front or behind the current car pose)
        '''
        nearest_wp_idx = self.waypoint_tree.query([
            self.pose.pose.position.x,
            self.pose.pose.position.y
        ],1)[1]
        closest_pt = np.array(self.waypoint_xy[nearest_wp_idx])
        prev_pt = np.array(self.waypoint_xy[nearest_wp_idx-1]) # any chance of idx out of range?
        curr_pt = np.array([self.pose.pose.position.x, self.pose.pose.position.y])
        val = np.dot(closest_pt - prev_pt, curr_pt - closest_pt)

        if val > 0:
            nearest_wp_idx = (nearest_wp_idx + 1) % len(self.waypoint_xy)
        return nearest_wp_idx

    def publish_trajectory(self, idx):
        '''
        Publishes the LOOKAHEAD_WPS trajectory points of the base_waypoints to the final_waypoint topic.
        If the end of the base_waypoints are reached, the published trajectory gets shorter until there is
        no trajectory point left.
        '''
        lane = Lane()
        lane.header = self.all_waypoints.header
        base_waypoints = self.all_waypoints.waypoints[idx:idx+LOOKAHEAD_WPS]
        if self.latest_stop_line_idx == -1 or (self.latest_stop_line_idx >= idx+LOOKAHEAD_WPS_FOR_DECELERATION):
            lane.waypoints = base_waypoints
        else:
            # do the deceleration...
            lane.waypoints = self.decelerate_waypoints(base_waypoints, idx)
            #rospy.logerr("Velocity: {:2.1f}-{:2.1f}-{:2.1f}-{:2.1f}-{:2.1f}".format(lane.waypoints[0].twist.twist.linear.x, 
            #    lane.waypoints[12].twist.twist.linear.x,
            #    lane.waypoints[24].twist.twist.linear.x,
            #    lane.waypoints[36].twist.twist.linear.x,
            #    lane.waypoints[49].twist.twist.linear.x))

        self.final_waypoints_pub.publish(lane)

    def publish_trajectory_v2(self, idx):
        '''
        Publishes the trajectory based on whether it is required to stop or not. If there is a stop request
        (elf.latest_stop_line_idx is not -1), then a deceleration is being planned acc to the model given
        in the explanation of this class. 
        '''
        lane = Lane()
        lane.header = self.all_waypoints.header
        base_waypoints = self.all_waypoints.waypoints[idx:idx+LOOKAHEAD_WPS]
        if self.latest_stop_line_idx == -1 or (self.latest_stop_line_idx >= idx+LOOKAHEAD_WPS_FOR_DECELERATION):
            
            self.decelleration_start_pos_already_found = False
            lane.waypoints = base_waypoints

        else:
            if self.decelleration_start_pos_already_found == False:
                '''
                if we are not within the deceleration zone, but we just have entered it (this condition)
                then we need to identify where we need to start the decelleration and where it shall be
                finished.
                '''
                self.total_decel_len = self.distance(self.all_waypoints.waypoints, idx, self.latest_stop_line_idx) - STOPLINE_BUFFER
                if not self.total_decel_len < REMAINING_DIST_TOO_CLOSE_FOR_DECEL:  # assuming that we're traveling at full speed
                    self.decelleration_start_pos_already_found = True
                    #rospy.logerr("Decelleration started with: {:1.1f}m until stopline".format(self.total_decel_len))
                #else:
                    #rospy.logerr("Keep driving, too close to stopline")

            lane.waypoints = []
            for i, wp in enumerate(base_waypoints):
                curr_idx = idx + i
                remaining_dist = self.distance(self.all_waypoints.waypoints, curr_idx, self.latest_stop_line_idx) - STOPLINE_BUFFER
                vel_at_dist = self.velocity_model(remaining_dist ,self.total_decel_len, MAX_SPEED)
                #if i == 0:
                    #rospy.logerr("Remaining decelleration length: {:1.1f}, current speed: {:1.1f}".format(remaining_dist, vel_at_dist))
                p = Waypoint()
                p.pose = wp.pose
                p.twist.twist.linear.x = vel_at_dist
                lane.waypoints.append(p)


        self.final_waypoints_pub.publish(lane)

    def velocity_model(self, remaining_dist, total_dist, max_velocity):
        '''
        Returns the velocity for a distance ahead of the stopline.
        Uses remaining_dist and total_dist to identify how far the 
        car is away from the stopline in the normalized model. The
        resulting normalized velocity is then adapted to the desired
        max speed (max_velocity).
        '''
        x_interp = np.array([min(remaining_dist / total_dist, 1.)])
        y_interp = interpolate.splev(x_interp, self.vel_spline, der=0)
        if y_interp[0] > 1.:
            y_interp[0] = 1.

        vel = abs(y_interp[0] * max_velocity)
        if vel < 0.2:
            vel = 0.
        return vel

    def decelerate_waypoints(self, base_wp, closest_wp):
        '''
        Like the method "velocity_model()", but returns
        a linear deceleration. 
        '''
        temp = []
        #dist_of_car_log = 0.
        for i, wp in enumerate(base_wp):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.latest_stop_line_idx - closest_wp - 2, 0)
            dist = self.distance(base_wp, i, stop_idx)
            '''
            Note: Dist is approx 64 (m?) if the stopline is 100 trajectory points away
            ... and 32 m if trajectory point length is 50
            ... and 52 m if trajectory point length is 80
            At 0 m (actually before!) the car needs to stop.
            Car travels at 11.1 (m/s)
            '''
            vel = abs((dist)*11.1/32.0) # remove one meter from the actual distance to make the car stop earlier
            #rospy.logerr("{:1.1f}".format(vel))

            if vel < 0.2:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            #rospy.logerr("{:1.1f}".format(p.twist.twist.linear.x))
            temp.append(p)
        
        return temp

    def pose_cb(self, msg):
        '''
        Stores the current pose of the car in this instance
        '''
        if not self.pose:
            rospy.loginfo("Received first pose...")
        self.pose = msg

    def waypoints_cb(self, waypoints):
        '''
        Callback for the base way points, which are provided once after startup (To be verified!).
        The waypoints contain pose and twist, however, we need the pose only. x and y coordinates 
        of the trajectory is maintained in a KDTree to enable a quick access. 
        '''
        self.all_waypoints = waypoints
        self.waypoints_with_manipulated_velocity = waypoints
        self.waypoint_xy = list()
        if not self.waypoint_xy:
            for waypoint in waypoints.waypoints:
                self.waypoint_xy.append((waypoint.pose.pose.position.x, waypoint.pose.pose.position.y))
            self.waypoint_tree = KDTree(self.waypoint_xy)
            rospy.loginfo("All waypoints received...")

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
