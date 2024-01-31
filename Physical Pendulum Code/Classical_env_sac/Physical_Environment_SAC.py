import math as mt
import numpy as np
import time

class Inverted_Pendulum_Enviroment():
    def __init__(self, 
                 analog_input_pin_number,
                 direction_pin_number,
                 pwm_pin_number,
                 timestep_length = 0.06):

        '''
        current state vector contains:
        current_state[0] = cart position
        current_state[1] = cart velocity
        current_state[2] = sine pendulum angle
        current_state[3] = cosine pendulum angle
        current_state[4] = pendulum angular velocity
        current_state[5] = pendulum angle
        '''

        self.timestep_length = timestep_length
        self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, mt.pi]
        self.previous_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.position_threshold = 9

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
        
        time.sleep(0.05)

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
    
    def observe_state(self, serialInst):

        self.current_state[5] = self.calculate_angle()
        
        self.current_state[2] = mt.sin(self.current_state[5])
        self.current_state[3] = mt.cos(self.current_state[5])
        
        diff = (self.current_state[5] - self.previous_state[5])
        
        if (diff > 5):
            self.current_state[4] = ((mt.pi + self.previous_state[5]) + (mt.pi - self.current_state[5]))/self.timestep_length
        elif (diff < -5):   
            self.current_state[4] = ((mt.pi - self.previous_state[5]) + (mt.pi + self.current_state[5]))/self.timestep_length
        else:
            self.current_state[4] = (self.current_state[5] - self.previous_state[5])/self.timestep_length
            
        self.previous_state[5] = self.current_state[5]
        
        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))
            self.current_state[0] = cart_pos/100
            
        self.current_state[1] = (self.current_state[0] - self.previous_state[0])/self.timestep_length
        self.previous_state[0] = self.current_state[0]
        
        
    def step(self, 
             action,
             serialInst):
        
        self.take_action(action)
        

        self.observe_state(serialInst)
        

        termination = bool(self.current_state[0] < -(self.position_threshold)
                           or self.current_state[0] > self.position_threshold
                           or self.current_state[4] > 10
                           or self.current_state[4] < -10
                           )
        if not termination:
             reward = -((self.current_state[3])**2 + 0.1 * self.current_state[2]**2 + 0.001 * (action*12)**2 + 0.1*(self.current_state[0])**2)
             #reward = 1/2 * (1 - self.current_state[1]) 
             
        else:
            print('termm')
            reward = -5000
            # self.take_action(0)
            # termination = False
            # self.observe_state()

        return self.current_state[0:5], reward, termination
    

    def reset(self, cart_pos):
        
        
        self.current_state = [0, 0, 0, 0, 0, 0]
        self.current_state[0] = cart_pos
        self.current_state[5] = self.calculate_angle()
        self.current_state[2] = mt.sin(self.current_state[5])
        self.current_state[3] = mt.cos(self.current_state[5])

        return self.current_state[0:5]
        

