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

from AgentNetwork import ActorNetwork, CriticNetwork, SAC_Agent
from Physical_V1 import Inverted_Pendulum_Enviroment

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#Create training environment
enviroment = Inverted_Pendulum_Enviroment()
state_space = 5
action_space = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model params
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
    current_state = enviroment.reset()
    action_arr = []

    for current_time_step in range(max_time_steps):
        start = time.time()
        
        (action, log_probability) = Agent.action_selection(torch.FloatTensor(current_state))
        action = action.detach().cpu().numpy().item()
        print(action)

        enviroment.take_action(action)

        if Agent.memory.size() > 500:
            print("...")
            episodic_actor_loss = Agent.training()

        (next_state, reward, done) = enviroment.step(action)
        
        Agent.memory.save_tuple((current_state,
                                 action,
                                 reward,
                                 next_state,
                                 done))

        episodic_score += reward
        current_state = next_state
        
        if done == True:
            break
        
        print("Current state: ", current_state)
        print("Rewards: ", reward)
        print("Cart pos is: ", current_state[0])
        print("Timesteps: ", current_time_step)
        action_arr.append(action)
        end = time.time()
        length_timestep = end - start #execution time
        print('Time execution: ', length_timestep)
        
    #Turn off motor
    enviroment.close()

    if episodic_score >= -350:
        torch.save(Agent.Actor, 'actor.pkl')
        torch.save(Agent.Critic_1_target, 'critic_1_target.pkl')
        torch.save(Agent.Critic_2_target, 'critic_2_target.pkl')
        torch.save(Agent.Critic_1, 'critic_1.pkl')
        torch.save(Agent.Critic_2, 'critic_2.pkl') 
       
    score.append(episodic_score)
    plt.plot(score)
    plt.show()
    if Agent.memory.size() > 500: 
        plt.plot(actor_loss)
        plt.show()
        actor_loss.append(episodic_actor_loss)
        print("Episode:{}, Score:{:.2f}, Actor Loss:{:.3f}".format(current_episode+1, 
                                                                   episodic_score, 
                                                                   episodic_actor_loss))

print("Finish Training:...")
