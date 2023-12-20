## This script uses SamplerQNN, 1 CNN

import gym
import sys


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
from PIL import Image
from matplotlib import pyplot, animation

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import qiskit_machine_learning
from qiskit_machine_learning.neural_networks import SamplerQNN, CircuitQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import ParameterVector
from qiskit.visualization.bloch import Bloch
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridActorNetwork(nn.Module):
    def __init__(self,
                 state_space,
                 action_space,
                 number_of_qubits = 1,
                 number_of_shots = 1024):
        super(HybridActorNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.number_of_qubits = number_of_qubits
        self.number_of_shots = number_of_shots

        self.qubits = [i for i in range(0, self.number_of_qubits)]
        self.quantum_circuit = qiskit.QuantumCircuit(self.number_of_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.beta = qiskit.circuit.ParameterVector('beta', self.state_space)

        self.input_layer()
        self.parameterized_quantum_circuit()
        quantum_neural_network = SamplerQNN(circuit = self.quantum_circuit,
                                              input_params = self.beta.params,
                                              weight_params = [self.theta],
                                              input_gradients = True)
        self.quantum_layer = TorchConnector(quantum_neural_network)

        # self.classical_layer_1 = nn.Linear(64, 128)
        # self.classical_layer_2 = nn.Linear(128, 256)
        self.classical_layer_3 = nn.Linear(64, self.action_space)

    def input_layer(self):
        self.quantum_circuit.h(self.qubits)
        self.quantum_circuit.barrier()

        self.quantum_circuit.rz(self.beta[0], self.qubits)
        self.quantum_circuit.ry(self.beta[1], self.qubits)
        self.quantum_circuit.rz(self.beta[2], self.qubits)
        self.quantum_circuit.barrier()

    def parameterized_quantum_circuit(self):
        self.quantum_circuit.rx(self.theta, self.qubits)
        self.quantum_circuit.barrier()

        print(self.quantum_circuit)

    def forward(self, current_state):
        output_layer_1 = self.quantum_layer(current_state)
        
        output_layer_1 = 1*output_layer_1[0] + (-1)*output_layer_1[1]
        output_layer_1 = output_layer_1.repeat(64)
        #output_layer_2 = F.relu(self.classical_layer_1(output_layer_1))
        #output_layer_3 = F.relu(self.classical_layer_2(output_layer_2))
        output_layer_4 = self.classical_layer_3(output_layer_1)

        policy = Categorical(F.softmax(output_layer_4, dim=-1))
        return policy

class HybridCriticNetwork(nn.Module):
    def __init__(self,
                 state_space,
                 action_space,
                 number_of_qubits = 1,
                 number_of_shots = 1024):
        super(HybridCriticNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.number_of_qubits = number_of_qubits
        self.number_of_shots = number_of_shots

        self.qubits = [i for i in range(0, self.number_of_qubits)]
        self.quantum_circuit = qiskit.QuantumCircuit(self.number_of_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.beta = qiskit.circuit.ParameterVector('beta', self.state_space)

        self.input_layer()
        self.parameterized_quantum_circuit()
        quantum_neural_network = SamplerQNN(circuit = self.quantum_circuit,
                                              input_params = self.beta.params,
                                              weight_params = [self.theta],
                                              input_gradients = True)
        self.quantum_layer = TorchConnector(quantum_neural_network)

        # self.classical_layer_1 = nn.Linear(64, 128)
        # self.classical_layer_2 = nn.Linear(128, 256)
        self.classical_layer_3 = nn.Linear(64, 1)

    def input_layer(self):
        self.quantum_circuit.h(self.qubits)
        self.quantum_circuit.barrier()

        self.quantum_circuit.rz(self.beta[0], self.qubits)
        self.quantum_circuit.ry(self.beta[1], self.qubits)
        self.quantum_circuit.rz(self.beta[2], self.qubits)
        self.quantum_circuit.barrier()

    def parameterized_quantum_circuit(self):
        self.quantum_circuit.rx(self.theta, self.qubits)
        self.quantum_circuit.barrier()

        

    def forward(self, current_state):
        output_layer_1 = self.quantum_layer(current_state)
        output_layer_1 = 1*output_layer_1[0] + (-1)*output_layer_1[1]
        output_layer_1 = output_layer_1.repeat(64)
        #output_layer_2 = F.relu(self.classical_layer_1(output_layer_1))
        #output_layer_3 = F.relu(self.classical_layer_2(output_layer_2))
        state_value = self.classical_layer_3(output_layer_1)

        return state_value

class QERL_Agent():
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
        #print(R)
        #print(self.rewards)
        discounted_rewards = []
        for step in reversed(range(len(self.rewards))):
            
            R = self.rewards[step] + self.discount_factor * R * self.done[step]
            discounted_rewards.insert(0, R)
            
        #print(discounted_rewards)
        
        return discounted_rewards

    def loss_function(self,
                      discounted_rewards):

        self.log_policies = torch.cat(self.log_policies)
        discounted_rewards = torch.cat(discounted_rewards).detach()
        self.state_values = torch.cat(self.state_values)

        advantage = discounted_rewards - self.state_values
        actor_loss = -(self.log_policies * advantage.detach()).mean()
        #print(self.log_policies)
        #print(advantage)
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

        for name, param in self.Actor.named_parameters():
            if param.requires_grad:
                #print(param.data)
                break
                
        
        return actor_loss.item(), critic_loss.item(), param.data.item()

def data_transformation(state):
        return [mt.atan(i) for i in state]

enviroment = gym.make("CartPole-v1").unwrapped

state_space = enviroment.observation_space.shape[0]-1
action_space = enviroment.action_space.n

actor_learning_rate = 0.04
critic_learning_rate = 0.04

max_time_steps = 500
number_of_epochs = 1
number_of_episodes = 500
discount_factor = 0.99
constant_for_average = 10

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

score = []
actor_loss = []
critic_loss = []

average_score = []
average_actor_loss = []
average_critic_loss = []

# First, we need to define the circuits:
theta_param = Parameter('θ')
phi_param =   Parameter('Φ')
beta1_param = Parameter('b1')
beta2_param = Parameter('b2')
beta3_param = Parameter('b3')
beta4_param = Parameter('b4')

#CHANGE GATES HERE
# Circuit A
qc_A = QuantumCircuit(1)
qc_A.h(0)

qc_A.rz(beta1_param, 0)
qc_A.ry(beta2_param, 0)
qc_A.rz(beta3_param, 0)
# Circuit B
qc_B = QuantumCircuit(1)
qc_B.h(0)

qc_B.rz(beta1_param, 0)
qc_B.ry(beta2_param, 0)
qc_B.rz(beta3_param, 0)

qc_B.rx(theta_param, 0)

# Bloch sphere plot formatting

b1 = Bloch()
b2 = Bloch()
b1.point_color = ['tab:blue']
b2.point_color = ['tab:blue']
b1.point_marker = ['o']
b2.point_marker = ['o']
b1.point_size=[2]
b2.point_size=[2]


def state_to_bloch(state_vec):
# Converts state vectors to points on the Bloch sphere
  phi = np.angle(state_vec.data[1])-np.angle(state_vec.data[0])
  theta = 2*np.arccos(np.abs(state_vec.data[0]))
  return [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]

obs1 = []
obs2 = []
obs3 = []
image_array = []
for current_episode in range(0, number_of_episodes):
    Agent.clear_memory()
    (current_state ) = enviroment.reset()
    episodic_score  = 0
    
    # Bloch sphere plot formatting   
    b1 = Bloch()
    b2 = Bloch()
    b1.point_color = ['tab:blue']
    b2.point_color = ['tab:blue']
    b1.point_marker = ['o']
    b2.point_marker = ['o']
    b1.point_size=[2]
    b2.point_size=[2]

    for time_step in range(0, max_time_steps):
        
        obs1.append(math.atan(current_state[1]))
        obs2.append(math.atan(current_state[2]))
        obs3.append(math.atan(current_state[3]))
        
        
        #enviroment.render()
        current_state = data_transformation(state = current_state[1:4])
        action, policy = Agent.action_selection(current_state)
        current_state = torch.FloatTensor(current_state).to(device)
        state_value = Critic(current_state)

        next_state, reward, done, info = enviroment.step(action)
        episodic_score += reward
        
        Agent.replay_memory(reward = reward,
                            policy = policy,
                            state_value = state_value,
                            done = done)

        current_state = next_state
        if done == True:
            break

    next_state = data_transformation(state = next_state[1:4])
    #print(next_state)
    next_state = torch.FloatTensor(next_state).to(device)
    next_state_value = Critic(next_state)
    #print(next_state_value)
    episodic_actor_loss, episodic_critic_loss, param = Agent.training(next_state_value = next_state_value)

    score.append(episodic_score)
    actor_loss.append(episodic_actor_loss)
    critic_loss.append(episodic_critic_loss)
    print('Episode:{} Score:{} Actor_Loss:{} Critic_Loss:{}'.format(current_episode,
                                                                    episodic_score,
                                                                    episodic_actor_loss,
                                                                    episodic_critic_loss))
    
    #Show states with parameter theta
    for n in range(len(obs1)):        
      state_2=Statevector.from_instruction(qc_B.bind_parameters({beta1_param:obs1[n], beta2_param:obs2[n], beta3_param:obs3[n], theta_param:param})) 
      b2.add_points(state_to_bloch(state_2))
    b2.show()
    image_array.append(b2)
    # obs1 = []
    # obs2 = []
    # obs3 = []

for i in range(0,int(len(score)/constant_for_average)):
    temp_score = sum(score[i * constant_for_average : (i+1) * constant_for_average])
    temp_actor_loss = sum(actor_loss[i * constant_for_average : (i+1) * constant_for_average])
    temp_critic_loss = sum(critic_loss[i * constant_for_average : (i+1) * constant_for_average])

    average_score.append(temp_score/10)
    average_actor_loss.append(temp_actor_loss/10)
    average_critic_loss.append(temp_critic_loss/10)

plt.plot(np.array(list(range(0, len(score)))), score, label='Score')
plt.plot(np.array(list(range(0, len(score), constant_for_average))), average_score, label='Average Score')
plt.legend()
plt.show()

plt.plot(np.array(list(range(0, len(actor_loss)))), actor_loss, label='Actor Loss')
plt.plot(np.array(list(range(0, len(actor_loss), constant_for_average))), average_actor_loss, label='Average Actor Loss')
plt.legend()
plt.show()

plt.plot(np.array(list(range(0, len(critic_loss)))), critic_loss, label='Critic Loss')
plt.plot(np.array(list(range(0, len(critic_loss), constant_for_average))), average_critic_loss, label='Average Critic Loss')
plt.legend()
plt.show()


for i in range(len(image_array)):
    filename = "bloch_" + str(i) + ".jpg"
    image_array[i].save('C:/Users/a1812176/Box/Code/Python Scripts/Pytorch script/qiskit-main/Qiskit/temp/' + filename )
    
 
