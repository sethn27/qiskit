import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
import time
import random

# Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])

# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer():
    def __init__(self, 
                 buffer_size, 
                 device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def save_tuple(self, tuple):
        self.buffer.append(tuple)

    def batch_sampling(self, 
               batch_size):
        batch = random.sample(self.buffer, 
                              batch_size)
        (current_state,
         action,
         reward,
         next_state,
         done) = ([], [], [], [], [])
        
        for tuple in batch:
            (temp_current_state,
             temp_action,
             temp_reward,
             temp_next_state,
             temp_done) = tuple
            current_state.append(temp_current_state)
            action.append(temp_action)
            reward.append([temp_reward])
            next_state.append(temp_next_state)
            done.append([1-int(temp_done)])

        current_state = torch.tensor(current_state,
                                     dtype=torch.float).to(self.device)
        action = torch.tensor(action,
                              dtype=torch.float).to(self.device)
        reward = torch.tensor(reward,
                              dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state,
                                  dtype=torch.float).to(self.device)
        done = torch.tensor(done,
                            dtype=torch.float).to(self.device)
        
        return current_state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)
    

class ActorNetwork(nn.Module):
    def __init__(self, 
                 state_space, 
                 action_space,
                 learning_rate):
        super(ActorNetwork, self).__init__()

        self.FC_layer_1 = nn.Linear(state_space, 64)
        self.FC_layer_2 = nn.Linear(64, 64)
        self.FC_mean = nn.Linear(64, action_space)
        self.FC_log_standard_deviation = nn.Linear(64, action_space)

        self.log_standard_deviation_min = -20
        self.log_standard_deviation_max = 2

        self.action_value_max = 1
        self.action_value_min = -1

        self.action_scale = (self.action_value_max - self.action_value_min)/2.0
        self.action_bias = (self.action_value_max + self.action_value_min)/2.0

        self.optimizer = optim.Adam(self.parameters(),
                                    lr = learning_rate)
        
    def forward(self,
                state):
        
        x = F.leaky_relu(self.FC_layer_1(state))
        x = F.leaky_relu(self.FC_layer_2(x))

        mean = self.FC_mean(x)
        log_standard_deviation = self.FC_log_standard_deviation(x)

        log_standard_deviation = torch.clamp(log_standard_deviation,
                                             self.log_standard_deviation_min,
                                             self.log_standard_deviation_max)
        
        (action,
         log_probability) = self.sample_action(mean,
                                               log_standard_deviation)
        
        return action, log_probability
    
    def sample_action(self,
                      mean,
                      log_standard_deviation):
        
        standard_deviation = torch.exp(log_standard_deviation)
        probability_distribution = Normal(mean,
                                          standard_deviation)
        
        sample = probability_distribution.rsample()
        x = torch.tanh(sample)

        action = self.action_scale * x + self.action_bias
        action = torch.clip(action, self.action_value_max, self.action_value_min)

        log_probability = probability_distribution.log_prob(sample)
        log_probability = log_probability - torch.sum(torch.log(self.action_scale * (1-x.pow(2)) + 1e-6),
                                                      dim = -1,
                                                      keepdim = True)
        
        return action, log_probability

class CriticNetwork(nn.Module):
    def __init__(self, 
                 state_space,
                 action_space,
                 learning_rate) -> None:
        super(CriticNetwork, self).__init__()

        self.FC_state = nn.Linear(state_space, 32)
        self.FC_action = nn.Linear(action_space, 32)

        self.FC_layer_1 = nn.Linear(64, 64)
        self.FC_layer_2 = nn.Linear(64, action_space)

        self.optimizer = optim.Adam(self.parameters(),
                                    lr = learning_rate)
        
    def forward(self,
                state,
                action):
        
        x = F.leaky_relu(self.FC_state(state))
        y = F.leaky_relu(self.FC_action(action))

        state_action = torch.cat([x, y], dim = -1)

        x = F.leaky_relu(self.FC_layer_1(state_action))
        stat_action_value = self.FC_layer_2(x)

        return stat_action_value
    
class SAC_Agent:
    def __init__(self,
                 state_space = 3,
                 action_space = 1,
                 actor_learning_rate = 0.001,
                 critic_learning_rate = 0.001,
                 batch_size = 200,
                 buffer_size = 100000,
                 discount_factor = 0.98,
                 tau = 0.005):
        
        self.state_space = state_space
        self.action_space = action_space
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_Factor = discount_factor
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau

        self.initial_alpha = 0.01
        self.target_entropy = -self.action_space  # == -1
        self.alpha_learning_rate = 0.005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(self.buffer_size, self.device)

        self.log_alpha = torch.tensor(np.log(self.initial_alpha)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], 
                                              lr=self.alpha_learning_rate)

        self.Actor = ActorNetwork(state_space = self.state_space,
                                  action_space = self.action_space,
                                  learning_rate = self.actor_learning_rate).to(self.device)
        self.Critic_1 = CriticNetwork(state_space = self.state_space,
                                      action_space = self.action_space,
                                      learning_rate = self.critic_learning_rate).to(self.device)
        self.Critic_1_target = CriticNetwork(state_space = self.state_space,
                                             action_space = self.action_space,
                                             learning_rate = self.critic_learning_rate).to(self.device)
        self.Critic_2 = CriticNetwork(state_space = self.state_space,
                                      action_space = self.action_space,
                                      learning_rate = self.critic_learning_rate).to(self.device)
        self.Critic_2_target = CriticNetwork(state_space = self.state_space,
                                             action_space = self.action_space,
                                             learning_rate = self.critic_learning_rate).to(self.device)

        self.Critic_1_target.load_state_dict(self.Critic_1.state_dict())
        self.Critic_2_target.load_state_dict(self.Critic_2.state_dict())

    def action_selection(self, state):
        with torch.no_grad():
            action, log_probability = self.Actor(state.to(self.device))
        return action, log_probability

    def compute_discouted_reward(self, batch):
        (current_state,
         action,
         reward,
         next_state,
         done) = batch
        
        with torch.no_grad():
            (next_state_action,
             next_state_log_probabilities) = self.Actor(next_state)
            entropy = - self.log_alpha.exp() * next_state_log_probabilities
            state_action_value_1 = self.Critic_1_target(next_state,
                                                        next_state_action)
            state_action_value_2 = self.Critic_2_target(next_state,
                                                        next_state_action)
            
            state_action_value = torch.min(state_action_value_1, state_action_value_2)
            discounted_reward = reward + self.discount_Factor * done * (state_action_value + entropy)

        return discounted_reward

    def training(self):

        batch = self.memory.batch_sampling(batch_size = self.batch_size)
        (current_state,
         action,
         reward,
         next_state,
         done) = batch

        discounted_reward = self.compute_discouted_reward(batch)

        critic_loss_1 = F.smooth_l1_loss(self.Critic_1(current_state,
                                                       action),
                                         discounted_reward)
        self.Critic_1.optimizer.zero_grad()
        critic_loss_1.mean().backward()
        self.Critic_1.optimizer.step()

        critic_loss_2 = F.smooth_l1_loss(self.Critic_2(current_state,
                                                       action),
                                         discounted_reward)
        self.Critic_2.optimizer.zero_grad()
        critic_loss_2.mean().backward()
        self.Critic_2.optimizer.step()

        predicted_action, log_probability = self.Actor(current_state)
        entropy = -self.log_alpha.exp() * log_probability

        state_action_value_1 = self.Critic_1(current_state, 
                                             predicted_action) 
        state_action_value_2 = self.Critic_2(current_state, 
                                             predicted_action)
        state_action_value = torch.min(state_action_value_1, state_action_value_2)
        actor_loss = -(state_action_value + entropy)  
        self.Actor.optimizer.zero_grad()
        actor_loss.mean().backward()
        self.Actor.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_probability + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        for param_target, param in zip(self.Critic_1_target.parameters(), self.Critic_1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Critic_2_target.parameters(), self.Critic_2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

        return actor_loss
