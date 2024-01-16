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
import math as mt
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board = pyfirmata2.Arduino('COM6')
board.samplingOn()

#it = pyfirmata2.util.Iterator(board)  
#it.start()

analog_input = board.get_pin('a:2:i')
pwm = board.get_pin('d:3:p')
brake = board.get_pin('d:9:o')
direction = board.get_pin('d:12:o')

a_chan = board.get_pin('d:4:u')
b_chan = board.get_pin('d:5:u')
  

pin_dir = 1

length =0.01 #timestep




def A_change(pulse): # #Interrupt A phase on encoder
        
        
        if  b_chan.read() == 0:
            if a_chan.read() == 0:
                invert_pen.red_pul(1) 
            else:
                invert_pen.add_pul(1)
        else:
            if a_chan.read() == 0:
                invert_pen.add_pul(1)
            else:
                invert_pen.red_pul(1)
        
    
def B_change(pulse): #Interrupt B phase on encoder
        
        
        if  a_chan.read() == 0:
            if b_chan.read() == 0:
                
                invert_pen.add_pul(1) 
            else:
                invert_pen.red_pul(1)
        else:
            if b_chan.read() == 0:
                
                invert_pen.red_pul(1)
            else:
                invert_pen.add_pul(1)
        
def analog_conversion():
    analog_value = analog_input.read()
    if analog_value != None:
        analog_value = analog_input.read()
        analog_value = round((analog_value*1024)*(2*math.pi/1024), 4)
        analog_value = round(math.pi/2 - analog_value, 4)
        print(analog_value)
        time.sleep(0.01)
    return analog_value #This is in radian
        
class environment:
    def __init__(self):
        self.pulse = 0 #encoder pulse
        self.reward = 0
        
        self.pen_angle = 0
        self.prev_angle = 0.0
        self.angular_vel = 0.0
        
        self.cart_pos = 0
        self.cart_vel = 0
        self.cart_prev = 0
        
    def add_pul(self, count):
        self.pulse = self.pulse + count
        return self.pulse
    def red_pul(self, count):
        self.pulse = self.pulse - count
        return self.pulse
    def choose_action(self, action):
        if action == 1:
            direction.write(1)
        else:
            direction.write(0)
    def vel_calc(self, pen_angle):
        
    
        self.pen_angle = pen_angle
        self.angular_vel = round((self.pen_angle - self.prev_angle)/length, 4)
        self.prev_angle = self.pen_angle
        
        self.cart_pos = invert_pen.pulse
        self.cart_vel = (self.cart_pos - self.cart_prev)/length
        self.cart_prev = self.cart_pos
        if (-0.2 <= pen_angle <= 0.2) :
            reward_count = 1
            invert_pen.reward = invert_pen.reward + 1
            if invert_pen.reward == 480:
                done =True
            else:
                done = False   
        else:
             reward_count = 0   
             
        print(reward_count)
        return self.cart_vel, self.pen_angle, self.angular_vel
            
        
#Create environment and callback function
invert_pen = environment()
a_chan.register_callback(A_change)
a_chan.enable_reporting()
b_chan.register_callback(B_change)
b_chan.enable_reporting()

#Model params

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
        return [mt.atan(i) for i in state]


    
for current_episode in range(0, number_of_episodes):
    Agent.clear_memory()
    
    timestep_count = 0
    #timestep = 0.01
    invert_pen.reward = 0
    
        
        
    print(invert_pen.pulse) #Encoder ticks
    while (invert_pen.pulse>10 or invert_pen.pulse<-10):
        brake.write(0)
        if invert_pen.pulse>10:
            direction.write(0)
            pwm.write(0.6)
        elif invert_pen.pulse<-10:
            direction.write(1)
            pwm.write(0.6)
        time.sleep(0.01)
        
    brake.write(1)        
    pwm.write(0)
        
    invert_pen.pulse = 0    
    print("3 sec wait")
    time.sleep(3)
    
    pen_angle = analog_conversion()
    current_state = [0, pen_angle, 0]
    print("Pen Angle", pen_angle)
    
    while ((-700 <= invert_pen.pulse <= 700) & (timestep_count < max_time_steps) & (-0.4 <= pen_angle <= 0.4)):
        start = time.time()
        
        current_state = data_transformation(state = current_state)
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)
        
        print(action)
        brake.write(0)
        pwm.write(0.8)
        #chosen_action = random.choice([0, 1])
        
        invert_pen.choose_action(action) #execute action here
        
        pen_angle = analog_conversion()
        next_state = invert_pen.vel_calc(pen_angle)  #get next state/ similar to step function
         
        Agent.replay_memory(reward = reward_count,
                            policy = policy,
                            state_value = state_value,
                            done = done)

        current_state = next_state
        
        
        
        #time.sleep(timestep)
        print(timestep_count)
        print("Reward: ",invert_pen.reward )
        
        end = time.time()
        length = end - start #execution time
        print('Time execution: ', length)
        print("Observations: ", current_state)
        timestep_count = timestep_count + 1
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
    
    score.append(invert_pen.reward)
    actor_loss.append(episodic_actor_loss)
    critic_loss.append(episodic_critic_loss)
    plt.plot(actor_loss)
    plt.show()
    print('Episode:{} Score:{} Actor_Loss:{} Critic_Loss:{}'.format(current_episode,
                                                                    invert_pen.reward,
                                                                    episodic_actor_loss,
                                                                    episodic_critic_loss))
    
board.exit()
    

    
    
    