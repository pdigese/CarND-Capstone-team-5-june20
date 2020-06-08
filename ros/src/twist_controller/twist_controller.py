from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, min_speed, max_speed, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, max_lon_decel, vehicle_mass, wheel_radius):    
        # FIXME: Find fitting parametrization for the longitudinal controller
        self.lon_ctrl = PID(kp = 0.2, 
            ki=0.001, 
            kd=2.0, 
            mn=min_speed, 
            mx=max_speed)
        self.lat_ctrl = YawController(wheel_base=wheel_base, 
            steer_ratio=steer_ratio, 
            min_speed=min_speed,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)
        self.t_past = None
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

    def control(self, lin_velocity_x, ang_velocity_z, curr_lin_velocity_x, curr_ang_velocity_z, dbw_enabled, t_now):
        '''
        TODO: Description of the input parameter
        lin_velocity_x:
        ...
        curr_time: uint32 nsec, uint32 sec (should be much more precise than float sec, let's hope there is some arhithmetic provided)
        '''
        throttle = 0.       # range 0...1 (no acc to may acceleration)
        brake = 0.          # brake force in Nm, higher value => stronger braking (only positive range)
        steer = 0.          # TODO: most likely in radian (rule: si-units everywhere...)
        max_brake_force = 400. # Nm

        steer = self.lat_ctrl.get_steering(lin_velocity_x, ang_velocity_z, curr_lin_velocity_x)

        if self.t_past is not None:
            t_d = t_now - self.t_past
        self.t_past = t_now

        if dbw_enabled is True and t_d is not None:
            # t delta between the last and current sample is required in order to integrate / differentiate properly
            vel_err = lin_velocity_x-curr_lin_velocity_x
            throttle = self.lon_ctrl.step(error=vel_err, sample_time=t_d)
            # a negative throttle must force the car to brake
            if curr_lin_velocity_x < 0.1 and lin_velocity_x == 0.:
                # do the 'handbrake'
                brake = max_brake_force
                throttle = 0.
            elif throttle < 0. and vel_err < 0.:
                # if there is 'negative throttle' => request for brake
                decelleration = max(abs(throttle), max_brake_force)
                brake = abs(self.vehicle_mass * decelleration * self.wheel_radius)
        else:
            # reset the internal I-value to prevent the I-part running amok when disengaged.  
            self.lon_ctrl.reset()

        return throttle, brake, steer
