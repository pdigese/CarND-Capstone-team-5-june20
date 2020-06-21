from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import math
import rospy # only for logging

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    '''
    Controller class which controls the longitudial movement
    - PI controller for throttle (implemented as PID, but D = 0)
    - P controller for brake
    and the lateral movement (feed forward controller).

    Implementation is based on Udacity's walkthrough video, with some additional
    changes for the collaboration of the throttle and brake controller.
    '''

    def __init__(self, min_speed, max_speed, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, max_lon_decel, vehicle_mass, wheel_radius):
        '''
        Constructor to initialize the longitudinal and lateral controller.
        Note that not all parameters are being used.
        ''' 
        self.lon_ctrl = PID(kp=0.5,
            ki=0.1,
            kd=0.,
            mn=-1., # min throttle and not max speed
            mx=0.7) # max throttle and not min speed
        self.lat_ctrl = YawController(wheel_base=wheel_base, 
            steer_ratio=steer_ratio,
            min_speed=min_speed,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)
        self.t_past = None
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.max_lon_decel = max_lon_decel

        f_sample = 50. # Hz sample frequency
        f_cutoff = 5. # Hz low pass first order
        self.lpf = LowPassFilter(tau=1./(2.*math.pi*f_cutoff), ts=1/f_sample)

    def control(self, lin_velocity_x, ang_velocity_z, curr_lin_velocity_x, curr_ang_velocity_z, dbw_enabled, t_now):
        '''
        Method executes the actual controller implementation. It needs to be called cyclically and fed with the latest
        requested and measured linear velocity and yaw rate, timestamped with t_now (Since the input signals are not
        provided with timestamp, there is a huge uncertainty. However, the difference between t_now and t_now-1 should 
        somehow contain the difference between the sample time points of the signals).
        '''
        throttle = 0.       # range 0...1 (no acc to may acceleration)
        brake = 0.          # brake force in Nm, higher value => stronger braking (only positive range)
        steer = 0.          # in radian (rule: si-units everywhere...)
        max_brake_force = 800. # Nm

        # there is some jitter in the measured velocity, therefore it needs to be filtered.
        curr_lin_velocity_x = self.lpf.filt(curr_lin_velocity_x)
        # keep using the steer controller even if dbw is enabled (controller does not have internal states,
        # so nothing can get messed up).
        steer = self.lat_ctrl.get_steering(lin_velocity_x, ang_velocity_z, curr_lin_velocity_x)

        t_d = None
        if self.t_past is not None:
            t_d = t_now - self.t_past
        self.t_past = t_now

        if dbw_enabled is True and t_d is not None:
            # t delta between the last and current sample is required in order to integrate / differentiate properly
            vel_err = lin_velocity_x-curr_lin_velocity_x
            throttle = self.lon_ctrl.step(error=vel_err, sample_time=t_d)

            if curr_lin_velocity_x < 0.1 and lin_velocity_x < 0.1:
                '''
                If there is more or less no request for velocity, it is assumed that the vehicle shall
                stop. Therefore use the handbrake if the vehicle reaches the actual standstill 
                (defined by velocity < 0.1). This state can be left by requesting higher velocity.
                '''
                brake = max_brake_force
                throttle = 0.
            elif throttle < -0.05 and vel_err < 0.:
                '''
                If the difference between requested and actual velocity is negative ('vel_err < 0.'), then the vehicle must
                reduce is speed. Moreover, we wait for a negative throttle (controller should follow the request), otherwise
                the time between switching from acceleration to braking is more or less 0.

                Note 1: There corr_factor was introduced due to testig with the script "dbw_test.py". There, the car always seems
                to use much more brake force than the implementation here. The corr_factor does correct this difference.

                Note 2: Waiting for the throttle to become negative is a kind of danger since it is not clear how much the brake
                request is delayed.
                '''
                corr_factor = 2.5
                decelleration = min(abs(self.max_lon_decel), abs(vel_err))
                brake = abs(self.vehicle_mass * decelleration * self.wheel_radius) * corr_factor
                throttle = 0.
            elif throttle > 0.05 and vel_err > 0.:
                '''
                If the controller is planning to accelerate, just make sure there is no brake being used at the same time.
                '''
                brake = 0.
            else:
                '''
                If the throttle is somewhere between -0.05 and 0.05, then neither the brake nor the throttle should be 
                used to avoid using both at the same time.
                '''
                brake = 0.
                throttle = 0.

        else:
            # reset the internal I-value to prevent the I-part running amok when disengaged.
            self.lon_ctrl.reset()

        #rospy.logerr("Linear Velocity: {}".format(lin_velocity_x))
        #rospy.logerr("Angular Velocity: {}".format(ang_velocity_z))
        #rospy.logerr("Current Linear Velocity: {}".format(curr_lin_velocity_x))
        #rospy.logerr("Current Angular Velocity: {}".format(steer))
        #rospy.logerr("Throttle: {}".format(throttle))
        #rospy.logerr("Brake: {}".format(brake))

        return throttle, brake, steer
