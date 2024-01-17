import math as mt
import numpy as np

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
        self.position_threshold = 5.0

        self.analog_input_pin_number = analog_input_pin_number
        self.direction_pin_number = direction_pin_number
        self.a_channel_pin_number = a_channel_pin_number
        self.b_channel_pin_number = b_channel_pin_number

    def take_action(self,
                    action):
        self.direction.write(action)

    def calculate_angle(self):
        analog_value = self.analog_input_pin_number.read()

        if analog_value != None:
                angle = round((analog_value * 1024) * (2 * mt.pi/1024), 4)
                angle = round(mt.pi/2 - angle, 4)

        return angle

    def step(self, 
             action,
             timestep_length):
        
        self.take_action(action)
        self.current_state[1] = self.calculate_angle()
        self.current_state[2] = round((self.current_state[1] - self.previous_state[1])/timestep_length, 4)

        self.current_state[0] = self.pulse
        self.current_state[3] = round((self.current_state[0] - self.previous_state[0])/timestep_length, 4)

        self.previous_state = self.current_state

        termination = bool(self.pulse < -self.position_threshold
                           or self.pulse > self.position_threshold)
        if not termination:
             reward = (1/2) * (1 - mt.cos(self.current_state[1]))

        else:
             reward = 0

        return self.current_state, reward, termination

    def reset(self):
        self.current_state = [0.0, 0.0, 0.0, 0.0]
        self.pulse = 0

        return self.current_state
    
    def add_pulse(self, count):
        self.pulse = self.pulse + count
        return self.pulse
    
    def reduce_pulse(self, count):
        self.pulse = self.pulse - count
        return self.pulse
    
    def A_change(self): # #Interrupt A phase on encoder
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
        
    
    def B_change(self): #Interrupt B phase on encoder

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