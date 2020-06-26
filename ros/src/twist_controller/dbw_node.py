#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        # provided by the twist topic callbacks
        self.dbw_enabled = None
        self.prev_dbw_enabled = None
        self.lin_velocity_x_cmd = None
        self.ang_velocity_yaw_cmd = None
        self.lin_velocity_x_feedback = None
        self.ang_velocity_yaw_feedback = None

        self.curr_throttle = None
        self.curr_brake = None
        self.curr_steer = None

        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -20.) # needs to be strong since we hace a short trajectory
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        # FIXME: which values should be used here:

        min_speed = 0.1     # m/s ? Why is it even required?
        max_speed = 20      # m/s ? What's the maximum allowed speed here?

        # TODO: Standard parameter are provided! Any ambition to make it more advanced?
        self.controller = Controller(min_speed=min_speed,
            max_speed=max_speed,
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle,
            max_lon_decel=decel_limit,
            vehicle_mass=vehicle_mass,
            wheel_radius=wheel_radius)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.sub_dbw_en = rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        self.sub_ctrl = rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)         # control movement  ("soll")
        self.sub_fb = rospy.Subscriber('/current_velocity', TwistStamped, self.car_curr_vel_cb) # feedback movement ("ist")

        self.loop()

    def twist_cmd_cb(self, msg):
        '''
        TODO: Describe why only the x velocity and yaw rate shall be used.
        '''
        # for debugging purposes:
        if self.lin_velocity_x_cmd is None:
            rospy.loginfo("First twist command received...")
        self.lin_velocity_x_cmd = msg.twist.linear.x
        self.ang_velocity_yaw_cmd = msg.twist.angular.z
    

    def dbw_enabled_cb(self, msg):
        '''
        Callback for the "engagement handler", which reports if there was a manual intervention
        (in simulator: 'manual' is checked).
        '''
        self.dbw_enabled = msg.data
        # for debugging purposes:
        if self.dbw_enabled != self.prev_dbw_enabled:
            rospy.loginfo("Drive by wire enable is {}".format(self.dbw_enabled))
        self.prev_dbw_enabled = msg.data

    def car_curr_vel_cb(self, msg):
        '''
        Ego motion of the car measured as a feedback for the steering.
        TODO: Describe why only the x velocity and yaw rate shall be used.
        '''
        # for debugging purposes:
        if self.lin_velocity_x_feedback is None:
            rospy.loginfo("First vehicle speed feedback received...")
        self.lin_velocity_x_feedback = msg.twist.linear.x
        self.ang_velocity_yaw_feedback = msg.twist.angular.z

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # Run the motion control only if required data is available
            if self.lin_velocity_x_cmd is not None and \
               self.lin_velocity_x_feedback is not None and \
               self.ang_velocity_yaw_cmd is not None and \
               self.ang_velocity_yaw_feedback is not None:
                self.curr_throttle, self.curr_brake, self.curr_steer = self.controller.control(lin_velocity_x=self.lin_velocity_x_cmd,
                                                                ang_velocity_z=self.ang_velocity_yaw_cmd,
                                                                curr_lin_velocity_x=self.lin_velocity_x_feedback,
                                                                curr_ang_velocity_z=self.ang_velocity_yaw_feedback,
                                                                dbw_enabled=self.dbw_enabled,
                                                                t_now=rospy.get_time()) # TODO: any other argument required?
            
            # TODO: The following publish "guards" might be a bit too much. Any changes required?
            if self.dbw_enabled and \
               self.curr_throttle is not None and \
               self.curr_brake is not None and \
               self.curr_steer is not None:
                self.publish(self.curr_throttle, self.curr_brake, self.curr_steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
