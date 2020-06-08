#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.cyclic_traj_gen_and_publishing()

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
            wp_idx = self.get_closest_waypoint_idx()
            self.publish_trajectory(wp_idx)
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
        curr_pt = np.array(self.pose.pose.position.x, self.pose.pose.position.y)
        val = np.dot(closest_pt - prev_pt, curr_pt - closest_pt)
        '''
        TODO: Find your own approach to identify the closest point...
        '''
        if val > 0:
            nearest_wp_idx = (nearest_wp_idx + 1) % len(nearest_wp_idx)
        return nearest_wp_idx

    def publish_trajectory(self, idx):
        lane = Lane()
        lane.header  = self.all_waypoints.header
        lane.waypoints = self.all_waypoints.waypoints[idx:idx+LOOKAHEAD_WPS]    # what happens if idx becomes to large? will is start from the beginning on again?
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        '''
        Stores the current pose of the car in this instance
        '''
        self.pose = msg

    def waypoints_cb(self, waypoints):
        '''
        Callback for the base way points, which are provided once after startup (To be verified!).
        The waypoints contain pose and twist, however, we need the pose only.
        '''
        self.all_waypoints = waypoints
        self.waypoint_xy = list()
        if not self.waypoint_xy:
            for waypoint in waypoints.waypoints:
                self.waypoint_xy.append((waypoint.pose.pose.position.x, waypoint.pose.pose.position.y))
            self.waypoint_tree = KDTree(self.waypoint_xy)
            rospy.loginfo("All waypoints received!")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Shall be implemented once the tl detection works
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. Shall be implemented once the obstacle detection works
        pass

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
