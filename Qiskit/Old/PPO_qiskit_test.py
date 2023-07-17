import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque

import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.distributions as distributions

from math import pi, sqrt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, assemble, Aer
from qiskit.tools.visualization import circuit_drawer, plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit import *
from numpy import linalg as la
from qiskit.tools.monitor import job_monitor
import qiskit.tools.jupyter

from ppo_agent_torch import Agent

        
        
class PPO_agent_qiskit():
    def __init__(self, action_space, state_space) -> None:
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space[0]
        self.e = 0.1  # Policy distance
        
        #self.qubits = [cirq.GridQubit(0, i) for i in range(1)]
        
        self.gamma = 0.98  # Discount factor
        self.K = 4  # Number of epochs
        self.T = 500  # Horizon
        self.M = 64  # Batch size
        self.ent = 0.1
        self.states = np.zeros((self.T, self.state_space))
        self.rewards = np.zeros((self.T, 1))
        self.actions = np.zeros((self.T, 1))
        self.probs = np.zeros((self.T, self.action_space))
        self.iter = 0
        
    def get_action(self, observation):
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            

            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()

            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            value = T.squeeze(value).item()

            return action, probs, value
        
    def remember(self, state, reward, action, probs):
        
        i = self.iter
        self.states[i] = state
        self.rewards[i] = reward
        self.actions[i] = action
        self.probs[i] = probs
        self.iter += 1
        
    
    def convert_data(self, x, flag=True):
        #ops = cirq.Circuit()
        
        #Create quantum circuit with 1 qubit
        ops = qiskit.QuantumCircuit(1,1)
        
        #Get gate angles fro observation:
        beta0 = math.atan(x[0])
        beta1 = math.atan(x[1])
        beta2 = math.atan(x[2])
        beta3 = math.atan(x[3])
        
          
        # ops.append([cirq.Moment([cirq.H(self.qubits[j]) for j in range(1)])])
        ops.h(0)
        
        ops.rz(beta1,0)
        ops.ry(beta2,0)
        ops.rz(beta1,0)
        
        return ops

    def train(self):
        batch_indices =[i for i in range(self.iter)]

        #Convert to tensor, PYTORCH CANNOT CONVERT QC TO TENSOR.
        state_batch = torch.tensor([self.convert_data(i, False) for i in self.states[batch_indices]])

        p_batch = torch.tensor(self.probs[:self.iter])
        
        action_batch = torch.tensor(self.actions[:self.iter])
        action_batch = [[i, action_batch[i][0]] for i in range(len(action_batch))]
        
        #p_batch = tf.cast(p_batch, dtype=tf.float32)
   
        #action_batch = tf.cast(action_batch, dtype=tf.int32)

        rewards = self.discount_reward(self.rewards[:self.iter])

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N=20
    batch_size = 5
    n_epochs =  4
    alpha = 0.0003
    agent = Agent(n_actions = env.action_space.n, batch_size=batch_size, alpha = alpha, n_epochs = n_epochs, input_dims = env.observation_space.shape)
    n_games = 2000
    
    
    
    ITERATIONS = 150
    windows = 20
    env = random.seed(34)

    env = gym.make("CartPole-v1")
    '''env.observation_space.shape'''
    
    agent = PPO_agent_qiskit(env.action_space.n, env.observation_space.shape)
    
    
    plot_rewards = []
    avg_reward = deque(maxlen=ITERATIONS)
    best_avg_reward = -math.inf
    rs = deque(maxlen=windows)
    
    for i in range(ITERATIONS):
        s1 = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # env.render()
            
            #Pick an action based on observations
            action, p = agent.get_action(s1)
            
            #Get the next observations and rewards by applying the action to the env
            s2, reward, done, info = env.step(action)
            episode_reward += reward
            
            #Remember last observations and actions
            agent.remember(s1, reward, action, p)
            s1 = s2


        agent.train()
        
        plot_rewards.append(episode_reward)
        rs.append(episode_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        if i >= windows:
            if avg > best_avg_reward:
                best_avg_reward = avg

        print("\rEpisode {}/{} || Best average reward {}, Current Average {}, Current Iteration Reward {}".format(i,
                                                                                                                  ITERATIONS,
                                                                                                                  best_avg_reward,
                                                                                                                  avg,
                                                                                                                  episode_reward),
              )