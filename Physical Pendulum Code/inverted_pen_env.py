import math
import numpy as np

class invert_pen_environment:
    def __init__(self):
        self.pulse = 0 #encoder pulse
        self.reward = 0
        
        self.pen_angle = 0
        self.prev_angle = 0.0
        self.angular_vel = 0.0
        
        self.cart_pos = 0
        self.cart_vel = 0
        self.cart_prev = 0
        
        self.cart_pos_threshold = 5.0
        self.pen_angle_threshold = 0.21
        self.state = None
        
    def add_pul(self, count):
        self.pulse = self.pulse + count
        return self.pulse
    def red_pul(self, count):
        self.pulse = self.pulse - count
        return self.pulse
    def choose_action(self, action, direction):
        if action == 1:
            direction.write(1)
        else:
            direction.write(0)
            
    def analog_conversion(self, analog_input):
            analog_value = analog_input.read()
            if analog_value != None:
                
                analog_value = round((analog_value*1024)*(2*math.pi/1024), 4)
                analog_value = round(math.pi/2 - analog_value, 4)
                
            return analog_value #This is in radian
        
    def step(self, analog_input, timestep_length):
        
        self.pen_angle = self.analog_conversion(analog_input)
        self.angular_vel = round((self.pen_angle - self.prev_angle)/timestep_length, 4)
        self.prev_angle = self.pen_angle
        
        self.cart_pos = self.pulse
        self.cart_vel = (self.cart_pos - self.cart_prev)/timestep_length
        self.cart_prev = self.cart_pos
        
        terminated = bool(
            self.pulse < -self.cart_pos_threshold
            or self.pulse > self.cart_pos_threshold
            #or self.pen_angle < - self.pen_angle_threshold
            #or self.pen_angle > self.pen_angle_threshold
            )
        
        if not terminated:
            reward_count = 1/2 - (1-math.cos(self.pen_angle))
            self.reward = self.reward + reward_count
        else:
            reward_count = 0
        self.state = (self.cart_vel, self.pen_angle, self.angular_vel)
        return np.array(self.state, dtype = np.float32), reward_count, terminated
    
    def reset(self, analog_input):
        self.pen_angle = self.analog_conversion(analog_input)
        self.pulse = 0
        self.reward = 0
        self.state = (0, self.pen_angle, 0)
        
        return np.array(self.state, dtype = np.float32)
             

        
            
