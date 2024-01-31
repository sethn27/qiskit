import math as mt
import numpy as np
import time
import pyfirmata2
import serial.tools.list_ports

class Inverted_Pendulum_Enviroment():
    def __init__(self,
                 timestep_length = 0.05):
        
        (self.analog_input_pin_number, 
         self.direction_pin_number, 
         self.pwm_pin_number,
         self.brake) = self.initialize_hardware()

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
        '''self.current_state = [0.0, 0.0, 0.0, 0.0, 0.0, mt.pi]
        self.previous_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'''
        self.position_threshold = 9

    def initialize_hardware(self):
        #Serial
        ports = serial.tools.list_ports.comports()
        self.serialInst = serial.Serial()

        portsList = []

        for onePort in ports:
            portsList.append(str(onePort))
            print(str(onePort))

        val = input("Select Port: COM")

        for x in range(0,len(portsList)):
            if portsList[x].startswith("COM" + str(val)):
                portVar = "COM" + str(val)
                print(portVar)

        self.serialInst.baudrate = 9600
        self.serialInst.port = portVar
        self.serialInst.open()

        #Arduino board:
        board = pyfirmata2.Arduino('COM6')
        board.samplingOn(20) #Sample every 25 ms

        #Pins definition
        analog_input = board.get_pin('a:2:i') #Pendulum
        pwm = board.get_pin('d:3:p') #Motor PWM (Speed)
        brake = board.get_pin('d:9:o') #Motor brake
        direction = board.get_pin('d:12:o') #Motor direction

        return analog_input, direction, pwm, brake

    def take_action(self,
                    action):
        
        self.brake.write(0)
        if action > 0:
            self.pwm_pin_number.write(action)
            self.direction_pin_number.write(1)
        elif action < 0:
            self.pwm_pin_number.write(-action)
            self.direction_pin_number.write(0)
        else:
            self.pwm_pin_number.write(0)
        
        #time.sleep(0.05)

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

        (position ,_ , angle, _) = self.state
        
        new_angle = self.calculate_angle()
        if self.serialInst.in_waiting:
            packet = self.serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))
            new_position = cart_pos/100

        diff = (self.state[2] - angle)
        
        if (diff > 5):
            angular_velocity = ((mt.pi + angle) + (mt.pi - new_angle))/self.timestep_length
        elif (diff < -5):   
            angular_velocity = ((mt.pi - angle) + (mt.pi + new_angle))/self.timestep_length
        else:
            angular_velocity = (new_angle - angle)/self.timestep_length
            
        #self.previous_state[5] = self.current_state[5]
            
        cart_velocity = (new_position - position)/self.timestep_length
        position = new_position
        angle = new_angle
        self.state = (position , cart_velocity, angle, angular_velocity)
        #self.previous_state[0] = self.current_state[0]
        
        
    def step(self, action):
        
        #self.take_action(action)
        time.sleep(0.02)
        self.observe_state(self.serialInst)
        (position, _, angle, angular_velocity) = self.state

        termination = bool(position < -(self.position_threshold)
                           or position > self.position_threshold
                           or angular_velocity > 10
                           or angular_velocity < -10
                           )
        if not termination:
             reward = -((angle)**2 + 0.1 * angular_velocity**2 + 0.001 * (action*12)**2 + 0.1*(position)**2)
             #reward = 1/2 * (1 - self.current_state[1]) 
             
        else:
            print('Episode Terminated')
            reward = -5000
            # self.take_action(0)
            # termination = False
            # self.observe_state()

        return np.array(self.state, dtype = np.float32), reward, termination
    

    def reset(self):
        
        position = self.reset_position()
        cart_velocity = 0.0
        angle = self.calculate_angle()
        angular_velocity = 0.0
        self.state = (position , cart_velocity, angle, angular_velocity)

        return np.array(self.state, dtype = np.float32)
    
    def reset_position(self):
        if self.serialInst.in_waiting:
            packet = self.serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = int(packet.decode('utf').rstrip('\n'))

        #Bring pendulum back    
        print("Cart pos is: ", cart_pos) #Encoder ticks
        while (cart_pos>10 or cart_pos<-10):
        
            if self.serialInst.in_waiting>10:
                self.serialInst.flushInput()
            
            self.brake.write(0)
            if cart_pos>10:
                self.direction_pin_number.write(1)
                self.pwm_pin_number.write(0.55)
            elif cart_pos<-10:
                self.direction_pin_number.write(0)
                self.pwm_pin_number.write(0.55)
            
            
            time.sleep(0.01)    
            self.brake.write(1)        
            self.pwm_pin_number.write(0)

            if self.serialInst.in_waiting:
                packet = self.serialInst.readline()
                #print(packet.decode('utf').rstrip('\n'))
                cart_pos = int(packet.decode('utf').rstrip('\n'))

        if self.serialInst.in_waiting>10:
            self.serialInst.flushInput()
        
        self.brake.write(1)        
        self.pwm_pin_number.write(0)
        time.sleep(0.05)
    
        if self.serialInst.in_waiting:
            packet = self.serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = int(packet.decode('utf').rstrip('\n'))
        print("5 sec wait")
        print("Cart pos is: ", cart_pos) #Encoder ticks
        time.sleep(5)
    
    
        if self.serialInst.in_waiting>10:
            self.serialInst.flushInput()

        return cart_pos
    
    def close(self):
        self.brake.write(1)
        self.pwm_pin_number.write(0)