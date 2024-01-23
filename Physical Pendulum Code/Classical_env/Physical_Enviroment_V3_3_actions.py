import math as mt
import numpy as np
import time

class Inverted_Pendulum_Enviroment():
    def __init__(self, 
                 analog_input_pin_number,
                 direction_pin_number,
                 pwm_pin_number,
                 a_channel_pin_number,
                 b_channel_pin_number):
        self.pulse = 0.0

        '''
        current state vector contains:
        current_state[0] = sine pendulum angle
        current_state[1] = cosine pendulum angle
        current_state[2] = cart angular veloctity
        current_state[3] = pendulum angle
        '''

        self.current_state = [0.0, 0.0, 0.0, 0.0]
        self.previous_state = [0.0, 0.0, 0.0, 0.0]
        self.position_threshold = 900
        self.angle_threshold = 0.3

        self.analog_input_pin_number = analog_input_pin_number
        self.direction_pin_number = direction_pin_number
        self.pwm_pin_number = pwm_pin_number
        self.a_channel_pin_number = a_channel_pin_number
        self.b_channel_pin_number = b_channel_pin_number
        #Register callback function for encoder phases
        # self.a_channel_pin_number.register_callback(self.A_change)
        # self.a_channel_pin_number.enable_reporting()
        # self.b_channel_pin_number.register_callback(self.B_change)
        # self.b_channel_pin_number.enable_reporting()

    def take_action(self,
                    action):
        
        if action > 0:
            self.pwm_pin_number.write(0.7)
            self.direction_pin_number.write(1)
        elif action < 0:
            self.pwm_pin_number.write(0.7)
            self.direction_pin_number.write(0)
        else:
            self.pwm_pin_number.write(0)
        time.sleep(0.05)
        

    def calculate_angle(self):
        analog_value = self.analog_input_pin_number.read()

        if analog_value != None:
            angle = round((analog_value * 1024) * (2 * mt.pi/1024), 4)
            angle = round(angle, 4)
            if angle > mt.pi:
                angle = round(2*mt.pi - angle, 4)
        else:
            angle = 0
        return angle

    def step(self, 
             action,
             timestep_length,
             current_pulse):
        
        #self.take_action(action)
        if action == -1.5:
            action = -(0.7*12)
        elif action == 1.5:
            action = (0.7*12)
        
        self.current_state[3] = self.calculate_angle()
        
        self.current_state[0] = mt.sin(self.current_state[3])
        self.current_state[1] = mt.cos(self.current_state[3])
        self.current_state[2] = (self.current_state[3] - self.previous_state[3])/timestep_length
        
        
        self.previous_state[3] = self.current_state[3]
        
        termination = bool(current_pulse < -self.position_threshold
                           or current_pulse > self.position_threshold)
        if not termination:
             reward = -(self.current_state[3]**2 + 0.1 * self.current_state[2]**2 + 0.001 * action**2)

        else:
             reward = -100
             self.take_action(-action)
             termination = False
             #self.pwm_pin_number.write(0)

        return self.current_state[0:3], reward, termination

    def reset(self):
        self.current_state[3] = self.calculate_angle()
        
        self.previous_state[3] = self.current_state[3]
        
        time.sleep(0.05)
        
        self.current_state[3] = self.calculate_angle()
        self.current_state[0] = mt.sin(self.current_state[3])
        self.current_state[1] = mt.cos(self.current_state[3])
        self.current_state[2] = (self.current_state[3] - self.previous_state[3])/0.05
        
        #self.pulse = 0

        return self.current_state[0:3]
    
    def add_pulse(self, count):
        self.pulse = round(self.pulse + count,2)
        return self.pulse
    
    def reduce_pulse(self, count):
        self.pulse = round(self.pulse - count,2)
        return self.pulse
    
    def A_change(self, pulse): # #Interrupt A phase on encoder
        if  self.b_channel_pin_number.read() == 0:
            if self.a_channel_pin_number.read() == 0:
                self.reduce_pulse(0.01) 
            else:
                self.add_pulse(0.01)

        else:
            if self.a_channel_pin_number.read() == 0:
                self.add_pulse(0.01)
            else:
                self.reduce_pulse(0.01)
        
    
    def B_change(self, pulse): #Interrupt B phase on encoder

        if  self.a_channel_pin_number.read() == 0:
            if self.b_channel_pin_number.read() == 0:
                self.add_pulse(0.01) 
            else:
                self.reduce_pulse(0.01)

        else:
            if self.b_channel_pin_number.read() == 0:
                
                self.reduce_pulse(0.01)
            else:
                self.add_pulse(0.01)