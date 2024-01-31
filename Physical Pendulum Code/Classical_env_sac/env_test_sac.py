import pyfirmata2
import time
import sys
import random

import torch
from torch.autograd import Function
#from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
#from sklearn.preprocessing import normalize

import math
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports

from Agent_Networks import ActorNetwork, CriticNetwork, SAC_Agent
from Physical_Environment_SAC import Inverted_Pendulum_Enviroment

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
enviroment = Inverted_Pendulum_Enviroment(analog_input, direction, pwm)



#Model params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_space = 5
action_space = 1

actor_learning_rate = 0.0001
critic_learning_rate = 0.001 
max_time_steps = 2000
number_of_epochs = 1
number_of_episodes = 1000
discount_factor = 0.99
constant_for_average = 10
reward_count = 0
done = False
pulse = 0
batch_size = 200
buffer_size = 100000
tau = 0.005

Agent = SAC_Agent(state_space = state_space,
                  action_space = action_space,
                  batch_size = batch_size,
                  buffer_size = buffer_size,
                  discount_factor = discount_factor,
                  actor_learning_rate = actor_learning_rate,
                  critic_learning_rate = critic_learning_rate,
                  tau = tau)

score = []
actor_loss = []
print("Start Training:...")

prev_cart_pos = 0
cart_vel = 0
     
for current_episode in range(number_of_episodes):
    episodic_score = 0

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
            pwm.write(0.55)
        elif cart_pos<-10:
            direction.write(0)
            pwm.write(0.55)
            
            
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

    current_state = enviroment.reset(cart_pos)
    action_arr = []

    for current_time_step in range(max_time_steps):
        start = time.time()
        
        (action, log_probability) = Agent.action_selection(torch.FloatTensor(current_state))
 
        action = action.detach().cpu().numpy()
        brake.write(0)
        print(action)
        
        (next_state, reward, done) = enviroment.step(action.item(), serialInst)
        
        Agent.memory.save_tuple((current_state,
                                 action,
                                 reward,
                                 next_state,
                                 done))

        episodic_score += reward
        current_state = next_state

        if Agent.memory.size() > 500:
            print("...")
            episodic_actor_loss = Agent.training()
        
        if done == True:
            
            break
        
        print("Current state: ", current_state)
        print("Rewards: ", reward)
        print("Cart pos is: ", cart_pos)
        print("Timesteps: ", current_time_step)
        action_arr.append(action)
        end = time.time()
        length_timestep = end - start #execution time
        print('Time execution: ', length_timestep)
        # if serialInst.in_waiting>10:
        #     serialInst.flushInput()

    #Turn off motor
    brake.write(1)
    pwm.write(0)

    if episodic_score >= -200:
        torch.save(Agent.Actor, 'actor.pkl')
        torch.save(Agent.Critic_1_target, 'critic_1.pkl')
        torch.save(Agent.Critic_2_target, 'critic_2.pkl') 
       
    score.append(episodic_score)
    plt.plot(score)
    plt.show()
    if Agent.memory.size() > 500: 
        plt.plot(actor_loss)
        plt.show()
        actor_loss.append(episodic_actor_loss)
        print("Episode:{}, Score:{:.2f}, Actor Loss:{:.3f}".format(current_episode+1, episodic_score, episodic_actor_loss))

print("Finish Training:...")
