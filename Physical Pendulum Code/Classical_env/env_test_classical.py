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

from CNN_Agent import ActorNetwork, CriticNetwork, Classical_Agent
from Physical_Enviroment_V2 import Inverted_Pendulum_Enviroment


#Arduino board:
board = pyfirmata2.Arduino('COM6')
board.samplingOn(1) #Sample every 1 ms

#Pins definition
analog_input = board.get_pin('a:2:i') #Pendulum
pwm = board.get_pin('d:3:p') #Motor PWM (Speed)
brake = board.get_pin('d:9:o') #Motor brake
direction = board.get_pin('d:12:o') #Motor direction

a_channel_pin_number = board.get_pin('d:6:u') #A phase of motor encoder
b_channel_pin_number = board.get_pin('d:7:u') #B phase of motor encoder

#a_chan = board.get_pin('a:4:i')
#b_chan = board.get_pin('a:5:i')
#Create training environment
PEN_ENV = Inverted_Pendulum_Enviroment(analog_input, direction, a_channel_pin_number, b_channel_pin_number)
class pulse_calculator():
    def __innit__(self):
        self.pulse=0

    def add_pulse(self, count):
        
        self.pulse = self.pulse + count
        return pulse
    
    def reduce_pulse(self, count):
        self.pulse = round(self.pulse - count,2)
        return pulse
    
pulse_cal = pulse_calculator()  
pulse_cal.pulse = 0
  
def A_change( pulse): # #Interrupt A phase on encoder
    if  b_channel_pin_number.read() == 0:
        if a_channel_pin_number.read() == 0:
            pulse_cal.reduce_pulse( 0.01) 
        else:
            pulse_cal.add_pulse(0.01)

    else:
        if a_channel_pin_number.read() == 0:
            pulse_cal.add_pulse(0.01)
        else:
            pulse_cal.reduce_pulse( 0.01)
    

def B_change( pulse): #Interrupt B phase on encoder
    
    if  a_channel_pin_number.read() == 0:
        if b_channel_pin_number.read() == 0:
            pulse_cal.add_pulse(0.01)
        else:
            pulse_cal.reduce_pulse( 0.01)

    else:
        if b_channel_pin_number.read() == 0:
            
            pulse_cal.reduce_pulse( 0.01)
        else:
            pulse_cal.add_pulse(0.01)  
            
#Register callback function for encoder phases
a_channel_pin_number.register_callback(A_change)
a_channel_pin_number.enable_reporting()
b_channel_pin_number.register_callback(B_change)
b_channel_pin_number.enable_reporting()

while True:
    print(pulse_cal.pulse)
    time.sleep(0.1)

#Model params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_space = 3
action_space = 2

actor_learning_rate = 0.04
critic_learning_rate = 0.04

max_time_steps = 500
number_of_epochs = 1
number_of_episodes = 500
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
prev_pulse = 0.1
while True:
    current_pulse = pulse_cal.pulse
    
    value = PEN_ENV.calculate_angle()
    print(current_pulse)
    vel = (current_pulse - prev_pulse)/0.01
    time.sleep(0.01)
    prev_pulse = current_pulse
    

    
    


for current_episode in range(0, number_of_episodes):
    Agent.clear_memory()
    
    timestep_count = 0
    length_timestep = 0.02
    episodic_score  = 0
    
    #Bring pendulum back    
    print("Cart pos is: ", pulse_cal.pulse) #Encoder ticks
    while (pulse_cal.pulse>0.3 or pulse_cal.pulse<-0.3):
        brake.write(0)
        if pulse_cal.pulse>0.3:
            direction.write(0)
            pwm.write(0.7)
        elif pulse_cal.pulse<-0.3:
            direction.write(1)
            pwm.write(0.7)
        time.sleep(0.005)    
    brake.write(1)        
    pwm.write(0)
 
    print("5 sec wait")
    print("Cart pos is: ", pulse_cal.pulse) #Encoder ticks
    time.sleep(3)
    
    current_state = PEN_ENV.reset()
 
    for timestep_count in range(0, max_time_steps):
        start = time.time()
        
        
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)
        
        brake.write(0)
        pwm.write(0.7)
        PEN_ENV.take_action(action)
        time.sleep(0.01)
        next_state, reward_count, done = PEN_ENV.step(action, length_timestep, pulse_cal.pulse)
        
        Agent.replay_memory(reward = reward_count,
                            policy = policy,
                            state_value = state_value,
                            done = done)
        
        episodic_score += reward_count
        current_state = next_state
        
        
        if done == True:
            break
        
        print("Current state: ", current_state)
        print("Rewards: ", episodic_score)
        print("Cart pos is: ", pulse_cal.pulse)
        
        action_arr.append(action)
        end = time.time()
        length_timestep = end - start #execution time
        print('Time execution: ', length_timestep)
        
    
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
        


 
    






