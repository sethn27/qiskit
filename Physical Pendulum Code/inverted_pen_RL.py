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

from QRL_Agent import HybridActorNetwork, HybridCriticNetwork, QERL_Agent
from inverted_pen_env import invert_pen_environment


#Arduino board:
board = pyfirmata2.Arduino('COM6')
board.samplingOn()

#Pins definition
analog_input = board.get_pin('a:2:i') #Pendulum
pwm = board.get_pin('d:3:p') #Motor PWM (Speed)
brake = board.get_pin('d:9:o') #Motor brake
direction = board.get_pin('d:12:o') #Motor direction

a_chan = board.get_pin('d:4:u') #A phase of motor encoder
b_chan = board.get_pin('d:5:u') #B phase of motor encoder


#Callback functions
def A_change(pulse): # #Interrupt A phase on encoder
        if  b_chan.read() == 0:
            if a_chan.read() == 0:
                PEN_ENV.red_pul(0.01) 
            else:
                PEN_ENV.add_pul(0.01)
        else:
            if a_chan.read() == 0:
                PEN_ENV.add_pul(0.01)
            else:
                PEN_ENV.red_pul(0.01)
        
    
def B_change(pulse): #Interrupt B phase on encoder

        if  a_chan.read() == 0:
            if b_chan.read() == 0:
                
                PEN_ENV.add_pul(0.01) 
            else:
                PEN_ENV.red_pul(0.01)
        else:
            if b_chan.read() == 0:
                
                PEN_ENV.red_pul(0.01)
            else:
                PEN_ENV.add_pul(0.01)
    
#Register callback function for encoder phases
a_chan.register_callback(A_change)
a_chan.enable_reporting()
b_chan.register_callback(B_change)
b_chan.enable_reporting()

#Create training environment
PEN_ENV = invert_pen_environment()

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

score = []
actor_loss = []
critic_loss = []

Actor = HybridActorNetwork(state_space = state_space,
                     action_space = action_space).to(device)
Critic = HybridCriticNetwork(state_space = state_space,
                       action_space = action_space).to(device)
Agent = QERL_Agent(Actor = Actor,
                        Critic = Critic,
                        max_time_steps = max_time_steps,
                        discount_factor = discount_factor,
                        actor_learning_rate = actor_learning_rate,
                        critic_learning_rate = critic_learning_rate,
                        number_of_epochs = number_of_epochs)

def data_transformation(state):
        return [math.atan(i) for i in state]
   
for current_episode in range(0, number_of_episodes):
    Agent.clear_memory()
    
    timestep_count = 0
    length_timestep = 0.02
    #timestep = 0.01
    PEN_ENV.reward = 0
    
        
    #Bring pendulum back    
    print("Cart pos is: ", PEN_ENV.pulse) #Encoder ticks
    while (PEN_ENV.pulse>1 or PEN_ENV.pulse<-1):
        brake.write(0)
        if PEN_ENV.pulse>1:
            direction.write(0)
            pwm.write(0.6)
        elif PEN_ENV.pulse<-1:
            direction.write(1)
            pwm.write(0.6)
        
        time.sleep(0.01)
        
    brake.write(1)        
    pwm.write(0)
        
      
    print("3 sec wait")
    time.sleep(3)
    
    current_state = PEN_ENV.reset(analog_input)
    
    for timestep_count in range(0, max_time_steps):
        start = time.time()
        
        current_state = data_transformation(state = current_state)
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)
        
        print(action)
        brake.write(0)
        pwm.write(0.8)
        
        PEN_ENV.choose_action(action, direction)
        
        next_state, reward_count, done = PEN_ENV.step(analog_input, length_timestep)
        
        Agent.replay_memory(reward = reward_count,
                            policy = policy,
                            state_value = state_value,
                            done = done)

        current_state = next_state
        print("Current state: ", current_state)
        print("Rewards: ", PEN_ENV.reward)
        
        time.sleep(0.02)
        end = time.time()
        length_timestep = end - start #execution time
        print('Time execution: ', length_timestep)
        
        
        if done == True:
            break
        
    #Turn off motor
    brake.write(1)
    pwm.write(0)
    next_state = data_transformation(state = next_state)
    #print(next_state)
    next_state = torch.FloatTensor(next_state).to(device)
    print(next_state)
    next_state_value = Critic(next_state)
    #print(next_state_value)
    episodic_actor_loss, episodic_critic_loss, param = Agent.training(next_state_value = next_state_value)
    
    score.append(PEN_ENV.reward)
    actor_loss.append(episodic_actor_loss)
    critic_loss.append(episodic_critic_loss)
    plt.plot(actor_loss)
    plt.show()
    print('Episode:{} Score:{} Actor_Loss:{} Critic_Loss:{}'.format(current_episode,
                                                                    PEN_ENV.reward,
                                                                    episodic_actor_loss,
                                                                    episodic_critic_loss))
    







