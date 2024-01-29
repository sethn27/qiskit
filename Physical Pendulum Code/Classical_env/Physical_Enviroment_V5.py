import math as mt
import numpy as np
import time

class Inverted_Pendulum_Enviroment():
    def __init__(self, 
                 analog_input_pin_number,
                 direction_pin_number,
                 pwm_pin_number,
                 timestep_length = 0.05):

        '''
        current state vector contains:
        current_state[0] = sine pendulum angle
        current_state[1] = cosine pendulum angle
        current_state[2] = pendulum angular veloctity
        current_state[3] = pendulum angle
        '''

        self.timestep_length = timestep_length
        self.current_state = [0.0, 0.0, 0.0, 0.0]
        self.previous_state = [0.0, 0.0, 0.0, 0.0]
        self.position_threshold = 900
        self.angle_threshold = 0.3

        self.analog_input_pin_number = analog_input_pin_number
        self.direction_pin_number = direction_pin_number
        self.pwm_pin_number = pwm_pin_number

    def take_action(self,
                    action):
        
        if action > 0:
            self.pwm_pin_number.write(action)
            self.direction_pin_number.write(1)
        elif action < 0:
            self.pwm_pin_number.write(-action)
            self.direction_pin_number.write(0)
        else:
            self.pwm_pin_number.write(0)
        
        time.sleep(self.timestep_length)

    def calculate_angle(self):
        analog_value = self.analog_input_pin_number.read()

        if analog_value != None:
            angle = round((analog_value * 1024) * (2 * mt.pi/1024), 4)
            angle = round(angle, 4)
            if angle > mt.pi:
                angle = round( angle - 2*(mt.pi), 4)
        else:
            angle = 0
        return angle
    
    def observe_state(self):

        self.current_state[3] = self.calculate_angle()
        
        self.current_state[0] = mt.sin(self.current_state[3])
        self.current_state[1] = mt.cos(self.current_state[3])
        
        diff = (self.current_state[3] - self.previous_state[3])
        if (diff > 5):
            self.current_state[2] = ((mt.pi - self.previous_state[3]) + (mt.pi + self.current_state[3]))/self.timestep_length
        elif (diff < -5):   
            self.current_state[2] = ((mt.pi + self.previous_state[3]) + (mt.pi - self.current_state[3]))/self.timestep_length
        else:
            self.current_state[2] = (self.current_state[3] - self.previous_state[3])/self.timestep_length
        self.previous_state[3] = self.current_state[3]
    
    def step(self, 
             action,
             current_pulse):
        
        self.take_action(action)
        '''if action == -1.5:
            action = -(0.7*12)
        elif action == 1.5:
            action = (0.7*12)'''

        self.observe_state()
        
        termination = bool(current_pulse < -(self.position_threshold)
                           or current_pulse > self.position_threshold
                           or self.current_state[2] > 10
                           or self.current_state[2] <-10)
        if not termination:
             reward = -((self.current_state[3])**2 + 0.1 * self.current_state[2]**2 + 0.001 * (action*12)**2)
             #reward = 1/2 * (1 - self.current_state[1]) - (current_pulse/800)**2
             print(reward)
        else:
            print('termm')
            reward = -500
            # self.take_action(0)
            # termination = False
            # self.observe_state()

        return self.current_state[0:3], reward, termination
    

    def reset(self):
        self.current_state[3] = self.calculate_angle()
        self.previous_state[3] = self.current_state[3]
        time.sleep(0.05)
        self.observe_state()
        
        #self.pulse = 0

        return self.current_state[0:3]
        