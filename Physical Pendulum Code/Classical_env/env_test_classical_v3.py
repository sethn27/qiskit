import pyfirmata2
import time
import sys
import random

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.preprocessing import normalize

import math
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import qiskit_machine_learning
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import ParameterVector
from qiskit.visualization.bloch import Bloch
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector

from CNN_Agent_3_action import ActorNetwork, CriticNetwork, Classical_Agent
from Physical_Enviroment_V3_3_actions import Inverted_Pendulum_Enviroment
import serial.tools.list_ports

#Serial
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portsList = []

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

val = input("Select Port: COM")

for x in range(0,len(portsList)):
    if portsList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()

#Arduino board:
board = pyfirmata2.Arduino('COM6')
board.samplingOn(20) #Sample every 25 ms

#Pins definition
analog_input = board.get_pin('a:2:i') #Pendulum
pwm = board.get_pin('d:3:p') #Motor PWM (Speed)
brake = board.get_pin('d:9:o') #Motor brake
direction = board.get_pin('d:12:o') #Motor direction

a_channel_pin_number = board.get_pin('d:5:u') #A phase of motor encoder
b_channel_pin_number = board.get_pin('d:6:u') #B phase of motor encoder

#a_chan = board.get_pin('a:4:i')
#b_chan = board.get_pin('a:5:i')
#Create training environment
PEN_ENV = Inverted_Pendulum_Enviroment(analog_input, direction, pwm, a_channel_pin_number, b_channel_pin_number)
class pulse_calculator():
    def __innit__(self):
        self.pulse=0

    def add_pulse(self, count):
        
        self.pulse = round(self.pulse + count,2)
        return pulse
    
    def reduce_pulse(self, count):
        self.pulse = round(self.pulse - count,2)
        return pulse
    
pulse_cal = pulse_calculator()  
pulse_cal.pulse = 0
  
def A_change(pulse): # #Interrupt A phase on encoder
       
    if  b_channel_pin_number.read() == 0:
        if a_channel_pin_number.read() == 0:
            pulse_cal.reduce_pulse( 0.01) 
        else:
            pulse_cal.add_pulse(0.01)

#     else:
#         if a_channel_pin_number.read() == 0:
#             pulse_cal.add_pulse(0.01)
#         else:
#             pulse_cal.reduce_pulse( 0.01)
    

# def B_change(pulse2): #Interrupt B phase on encoder
    
#     if  a_channel_pin_number.read() == 0:
#         if b_channel_pin_number.read() == 0:
#             pulse_cal.add_pulse(0.01)
#         else:
#             pulse_cal.reduce_pulse( 0.01)

#     else:
#         if b_channel_pin_number.read() == 0:
            
#             pulse_cal.reduce_pulse( 0.01)
#         else:
#             pulse_cal.add_pulse(0.01)  
            
#Register callback function for encoder phases
a_channel_pin_number.register_callback(A_change)
a_channel_pin_number.enable_reporting()
# b_channel_pin_number.register_callback(B_change)
# b_channel_pin_number.enable_reporting()


#Model params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_space = 4
action_space = 3

actor_learning_rate = 0.0005
critic_learning_rate = 0.005

max_time_steps = 200
number_of_epochs = 1
number_of_episodes = 5000
discount_factor = 0.99
constant_for_average = 10
reward_count = 0
done = False
pulse = 0

Actor = ActorNetwork(state_space = state_space,
                     action_space = action_space).to(device)
Critic = CriticNetwork(state_space = state_space,
                       action_space = action_space).to(device)
Agent = Classical_Agent(Actor = Actor,
                        Critic = Critic,
                        action_space = action_space,
                        max_time_steps = max_time_steps,
                        discount_factor = discount_factor,
                        actor_learning_rate = actor_learning_rate,
                        critic_learning_rate = critic_learning_rate,
                        number_of_epochs = number_of_epochs)

score = []
actor_loss = []
critic_loss = []
action_arr = []

average_score = []
average_actor_loss = []
average_critic_loss = []



    
for current_episode in range(0, number_of_episodes):
    Agent.clear_memory()
    
    timestep_count = 0
    length_timestep = 0.015
    episodic_score  = 0
    #while True:
    
    if serialInst.in_waiting:
        packet = serialInst.readline()
        #print(packet.decode('utf').rstrip('\n'))
        cart_pos = int(packet.decode('utf').rstrip('\n'))
        
    
    #Bring pendulum back    
    print("Cart pos is: ", cart_pos) #Encoder ticks
    while (cart_pos>50 or cart_pos<-50):
     
        brake.write(0)
        if cart_pos>10:
            direction.write(1)
            pwm.write(0.5)
        elif cart_pos<-10:
            direction.write(0)
            pwm.write(0.5)
            
        if serialInst.in_waiting>10:
            serialInst.flushInput()
            
        time.sleep(0.05)    
        brake.write(1)        
        pwm.write(0)
        
        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = int(packet.decode('utf').rstrip('\n'))
        
    print("5 sec wait")
    print("Cart pos is: ", cart_pos) #Encoder ticks
    time.sleep(5)
    pulse_cal.pulse = 0
    current_state = PEN_ENV.reset()
    action_arr = []
    if serialInst.in_waiting>10:
        serialInst.flushInput()
        
    for timestep_count in range(0, max_time_steps):
        start = time.time()
        
        
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)
        
        brake.write(0)
        #PEN_ENV.take_action(action)
        
        
        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))
            
        next_state, reward_count, done = PEN_ENV.step(action, length_timestep, cart_pos)

        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))

        next_state.append(cart_pos)
        
        Agent.replay_memory(reward = reward_count,
                            policy = policy,
                            state_value = state_value,
                            done = done)
        
        episodic_score += reward_count
        current_state = next_state
        
        
        if done == True:
            
            break
        
        print("Current state: ", current_state)
        #print("Rewards: ", episodic_score)
        print("Cart pos is: ", cart_pos)
        print("Timesteps: ", timestep_count)
        action_arr.append(action)
        end = time.time()
        length_timestep = end - start #execution time
        print('Time execution: ', length_timestep)
        if serialInst.in_waiting>10:
            serialInst.flushInput()
    
    #Turn off motor
    brake.write(1)
    pwm.write(0)
    
    next_state = torch.FloatTensor(next_state).to(device)
    next_state_value = Critic(next_state)
    episodic_actor_loss, episodic_critic_loss = Agent.training(next_state_value = next_state_value)
    
    score.append(episodic_score)
    actor_loss.append(episodic_actor_loss)
    critic_loss.append(episodic_critic_loss)
    plt.plot(actor_loss)
    plt.show()
    print('Episode:{} Score:{} Actor_Loss:{} Critic_Loss:{}'.format(current_episode,
                                                                    episodic_score,
                                                                    episodic_actor_loss,
                                                                    episodic_critic_loss))
    
    
    
    
    
    
    
    
    