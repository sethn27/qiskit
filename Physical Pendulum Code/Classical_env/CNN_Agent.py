import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.layer_1 = nn.Linear(self.state_space, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(128, self.action_space)

    def forward(self, current_state):
        output_layer_1 = F.relu(self.layer_1(current_state))
        #output_layer_2 = F.relu(self.layer_2(output_layer_1))
        output_layer_3 = self.layer_3(output_layer_1)
        policy = Categorical(F.softmax(output_layer_3, dim=-1))
        return policy

class CriticNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(CriticNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.layer_1 = nn.Linear(self.state_space, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(128, 1)

    def forward(self, current_state):
        output_layer_1 = F.relu(self.layer_1(current_state))
        #output_layer_2 = F.relu(self.layer_2(output_layer_1))
        state_value = self.layer_3(output_layer_1)
        return state_value

class Classical_Agent():
    def __init__(self,
                 Actor,
                 Critic,
                 max_time_steps,
                 actor_learning_rate,
                 critic_learning_rate,
                 discount_factor,
                 number_of_epochs):

        self.max_time_steps = max_time_steps
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.number_of_epochs = number_of_epochs

        self.Actor = Actor
        self.Critic = Critic

        self.actor_optimizer = optim.Adam(self.Actor.parameters(), self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), self.critic_learning_rate)

    def clear_memory(self):
        self.rewards = []
        self.state_values = []
        self.log_policies = []
        self.done = []

    def replay_memory(self,
                      state_value,
                      reward,
                      policy,
                      done):

        self.log_policies.append(policy)
        self.state_values.append(state_value)
        self.rewards.append(torch.tensor([reward],
                                         dtype = torch.float,
                                         device = device))
        self.done.append(torch.tensor([1-done],
                                      dtype = torch.float,
                                      device = device))

    def action_selection(self,
                         current_state):
        current_state = torch.FloatTensor(current_state).to(device)
        policy = self.Actor(current_state)
        action = policy.sample()
        policy = policy.log_prob(action).unsqueeze(0)

        return action.cpu().numpy(), policy

    def compute_discounted_rewards(self,
                                   next_state_value):
        R = next_state_value
        discounted_rewards = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.discount_factor * R * self.done[step]
            discounted_rewards.insert(0, R)
        return discounted_rewards

    def loss_function(self,
                      discounted_rewards):

        self.log_policies = torch.cat(self.log_policies)
        discounted_rewards = torch.cat(discounted_rewards).detach()
        self.state_values = torch.cat(self.state_values)

        advantage = discounted_rewards - self.state_values
        actor_loss = -(self.log_policies * advantage.detach()).mean()
        critic_loss = F.huber_loss(discounted_rewards, self.state_values)

        return actor_loss, critic_loss

    def training(self, next_state_value):
        discounted_rewards = self.compute_discounted_rewards(next_state_value)
        actor_loss, critic_loss = self.loss_function(discounted_rewards)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()