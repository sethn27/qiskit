import math as mt
import numpy as np
import time

class Inverted_Pendulum_Enviroment():
    def __init__(self, 
                 analog_input_pin_number,
                 direction_pin_number,
                 a_channel_pin_number,
                 b_channel_pin_number):
        self.pulse = 0.0

        '''
        current state vector contains cart velocity, angle, angular velocity, and position
        current_state[0] = cart velocity
        current_state[1] = cart angle
        current_state[2] = cart angular veloctity
        current_state[3] = cart position
        '''

        self.current_state = [0.0, 0.0, 0.0, 0.0]
        self.previous_state = [0.0, 0.0, 0.0, 0.0]
        self.position_threshold = 8.0
        self.angle_threshold = 0.3

        self.analog_input_pin_number = analog_input_pin_number
        self.direction_pin_number = direction_pin_number
        self.a_channel_pin_number = a_channel_pin_number
        self.b_channel_pin_number = b_channel_pin_number
        #Register callback function for encoder phases
        # self.a_channel_pin_number.register_callback(self.A_change)
        # self.a_channel_pin_number.enable_reporting()
        # self.b_channel_pin_number.register_callback(self.B_change)
        # self.b_channel_pin_number.enable_reporting()

    def take_action(self,
                    action):
        self.direction_pin_number.write(action)
        

    def calculate_angle(self):
        analog_value = self.analog_input_pin_number.read()

        if analog_value != None:
            angle = round((analog_value * 1024) * (2 * mt.pi/1024), 4)
            angle = round(angle- mt.pi/2 , 4)
        else:
            angle = 0
        return angle

    def step(self, 
             action,
             timestep_length,
             current_pulse):
        
        #self.take_action(action)
        
        self.current_state[1] = self.calculate_angle()
        self.current_state[2] = (self.current_state[1] - self.previous_state[1])/timestep_length
        
        self.current_state[3] = current_pulse
        self.current_state[0] = round((self.current_state[3] - self.previous_state[3])/timestep_length, 4)
        
        # self.previous_state = self.current_state
        self.previous_state[1] = self.current_state[1]
        self.previous_state[3] = self.current_state[3]
        print(self.current_state)
        termination = bool(current_pulse < -self.position_threshold
                           or current_pulse > self.position_threshold
                           or self.current_state[1] < mt.pi - self.angle_threshold
                           or self.current_state[1] > mt.pi + self.angle_threshold)
        if not termination:
             reward = 1

        else:
             reward = 0

        return self.current_state[0:3], reward, termination

    def reset(self):
        self.current_state = [0.0, 0.0, 0.0, 0.0]
        self.current_state[1] = self.calculate_angle()
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