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
from Physical_Enviroment_V4 import Inverted_Pendulum_Enviroment
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

#Create training environment
PEN_ENV = Inverted_Pendulum_Enviroment(analog_input, direction, pwm)



#Model params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_space = 4
action_space = 3

actor_learning_rate = 0.0005 #0.0005
critic_learning_rate = 0.005 #0.005

max_time_steps = 1000
number_of_epochs = 1
number_of_episodes = 3500
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

# while True:
#     print(PEN_ENV.calculate_angle())
#     time.sleep(0.1)
    
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
    while (cart_pos>10 or cart_pos<-10):
        
        if serialInst.in_waiting>10:
            serialInst.flushInput()
            
        brake.write(0)
        if cart_pos>10:
            direction.write(1)
            pwm.write(0.5)
        elif cart_pos<-10:
            direction.write(0)
            pwm.write(0.5)
            
            
        time.sleep(0.01)    
        brake.write(1)        
        pwm.write(0)
        
        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = int(packet.decode('utf').rstrip('\n'))
            
    if serialInst.in_waiting>10:
        serialInst.flushInput()
        
    brake.write(1)        
    pwm.write(0)
    time.sleep(0.05)
    
    if serialInst.in_waiting:
        packet = serialInst.readline()
        #print(packet.decode('utf').rstrip('\n'))
        cart_pos = int(packet.decode('utf').rstrip('\n'))
    print("5 sec wait")
    print("Cart pos is: ", cart_pos) #Encoder ticks
    time.sleep(5)
    
    
    if serialInst.in_waiting>10:
        serialInst.flushInput()
        
    current_state = PEN_ENV.reset()
    current_state.append(cart_pos/100)
    action_arr = []
        
    for timestep_count in range(0, max_time_steps):
        start = time.time()
        
        
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)
        
        brake.write(0)
        print(action)
        
        
        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))
            
        next_state, reward_count, done = PEN_ENV.step(action, cart_pos)

        if serialInst.in_waiting:
            packet = serialInst.readline()
            #print(packet.decode('utf').rstrip('\n'))
            cart_pos = float(packet.decode('utf').rstrip('\n'))

        next_state.append(cart_pos/100)
        
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
    
    if episodic_score >= 20:
        torch.save(Actor, 'actor.pkl')
        torch.save(Critic, 'critic.pkl')
    score.append(episodic_score)
    actor_loss.append(episodic_actor_loss)
    critic_loss.append(episodic_critic_loss)
    plt.plot(actor_loss)
    plt.show()
    plt.plot(score)
    plt.show()
    
    
    print('Episode:{} Score:{} Actor_Loss:{} Critic_Loss:{}'.format(current_episode,
                                                                    episodic_score,
                                                                    episodic_actor_loss,
                                                                    episodic_critic_loss))
    

np.savetxt('Actor.txt', actor_loss)
np.savetxt('Score.txt', score)
np.savetxt('Critic.txt', critic_loss)

    
    
    
    
    
    