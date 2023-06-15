import gym 
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import qiskit_machine_learning
from qiskit_machine_learning.neural_networks import SamplerQNN,CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector


class qrlPQC:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, shots):
        
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        
        self.theta = qiskit.circuit.Parameter('theta')
        self.beta = qiskit.circuit.ParameterVector('beta', 3)
        
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        
        self.circuit.rz(self.beta[0], all_qubits)
        self.circuit.ry(self.beta[1], all_qubits)
        self.circuit.rz(self.beta[2], all_qubits)
        self.circuit.barrier()
        
        self.circuit.rz(self.theta, all_qubits)
        
        
        self.circuit.measure_all()
        # ---------------------------

        self.shots = shots
    


class ExpectAndRepeatLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.number_of_reuse = 64
    
    def forward(self, x):
        exp_val = 0*x[0] + 1*x[1]
        return exp_val.repeat(self.number_of_reuse)

#Define qnn model:
circ = qrlPQC(1, 100)
qnn = SamplerQNN(
    circuit=circ.circuit,
    input_params=circ.beta.params,
    weight_params=[circ.theta],
    input_gradients=True
)

qlayer = TorchConnector(qnn)
    
customRepeat = ExpectAndRepeatLayer()

#Actor model, ReLU, optimizer???
model_actor = torch.nn.Sequential(qlayer, customRepeat, torch.nn.Linear(64, 2), torch.nn.Softmax(dim=0))
#Critic model
model_critic = torch.nn.Sequential(qlayer, customRepeat, torch.nn.Linear(64,1))


#Optimizer for parameter
optimizer_critic = optim.Adam(model_critic.parameters(), lr=0.04)
optimizer_actor = optim.Adam(model_actor.parameters(), lr=0.004)

#Use the Actor model to pick action, critic to estimate value
def pick_action(obs):
    
  tensor_obs = torch.Tensor(obs[1:4])     
  print(tensor_obs)
  #no_grad(), disable gradient
  with torch.no_grad():
      prob = model_actor(tensor_obs)
      value = model_critic(tensor_obs)
      
      action = torch.argmax(prob)
      # print("Output: " + str(prob))
      # print("Action: " + str(action))
      # print("Value: " + str(value))
      
      action = action.item() #Take action out of tensor
      
  return action, prob

class Agent(object):
    def __init__(self, action_space, state_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space[0]
        self.iter = 0
        
        self.H = 500  # Horizon, max number of element within tensor (max time step)
        self.states = np.zeros((self.H, self.state_space))
        self.rewards = np.zeros((self.H, 1))
        self.actions = np.zeros((self.H, 1))
        self.probs = np.zeros((self.H, self.action_space))
        
        self.gamma = 0.98  # Discount factor
        self.K = 1  # Number of epochs
        self.e = 0.1  # Policy distance
    
    def remember(self, state, reward, action, probs):
        
        i = self.iter
        self.states[i] = state
        self.rewards[i] = reward
        self.actions[i] = action
        self.probs[i] = probs
        self.iter += 1
        
    def discount_reward(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                Gt = 0
            else:
                Gt = rewards[i] + self.gamma * Gt
            d_rewards[i] = Gt
        return d_rewards
    
    def ppo_loss(self, cur_pol, old_pol, advantages):
        
        ratio = torch.div(cur_pol,old_pol).squeeze()
        print("ratio" +str(ratio))
        ratio = torch.clip(ratio, 1e-10, 10 - 1e-10)
        clipped = torch.clip(ratio, 1 - self.e, 1 + self.e).squeeze()
        print(clipped.shape)
        
        loss = -torch.mean(torch.min(advantages * ratio , advantages * clipped ))

        return loss
    
    def learn(self):
        # batch_indices = [i for i in range(self.iter)]
        # print("a" +str(batch_indices))
    
        state_batch = torch.Tensor(self.states[:self.iter])
        print(state_batch)
        
        p_batch = torch.Tensor(self.probs[:self.iter])
        p_batch = p_batch.double()
        print(p_batch)
        
        action_batch = torch.Tensor(self.actions[:self.iter])
        action_batch = action_batch.double()     
        print(action_batch)
        
        rewards = self.discount_reward(self.rewards[:self.iter]) #Calculate discounted reward sum
        rewards = torch.Tensor(rewards).squeeze()
        print(rewards)
        
        value_tensor = torch.zeros(state_batch.shape[0])
        actor_predict = torch.zeros((state_batch.shape[0],2))
        
        for _ in range(self.K):
            for i in range(state_batch.shape[0]):
                value_tensor[i] = model_critic(state_batch[i,1:4])                                      
                print(value_tensor)
            
            critic_loss = torch.mean((value_tensor - rewards)**2)
            print("critc loss" +str(critic_loss))
            
            
            optimizer_critic.zero_grad()
            critic_loss.backward() #gradient descent of critic loss
            optimizer_critic.step()
            
            
            
            advantage = rewards - value_tensor        
            advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8) #advantages normalized
            print("advantage: " +str(advantage))
            
            
            for i in range(state_batch.shape[0]):              
                actor_predict[i] = model_actor(state_batch[i,1:4])
            print(actor_predict)  
            print(p_batch)
            
            actor_loss = self.ppo_loss(actor_predict, p_batch, advantage.squeeze())
            
            optimizer_actor.zero_grad()
            actor_loss.backward() #gradient descent of critic loss
            optimizer_actor.step()
        
        
        self.iter = 0
            
#Load Environment

environment_name = "CartPole-v0"
env = gym.make(environment_name)
episodes = 1
PPO_agent = Agent(env.action_space.n, env.observation_space.shape)


for episode in range(1, episodes+1):
    ini_state = env.reset()    
    done = False
    score = 0 
    
    while not done:
        #env.render()       
        
        action, prob = pick_action(ini_state)
        print(action)
        n_state, reward, done, info = env.step(action)
        score+=reward
        PPO_agent.remember(ini_state, reward, action, prob)
        
        ini_state = n_state
    PPO_agent.learn()
    print('Episode:{} Score:{}'.format(episode, score))
#env.close()
