import gym 
import sys
import random
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import math
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import qiskit_machine_learning
from qiskit_machine_learning.neural_networks import SamplerQNN,CircuitQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import ParameterVector
from qiskit.visualization.bloch import Bloch
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector


class QNN_PPO_Agent(object):
    def __init__(self, action_space, state_space):
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space[0]
        self.iter = 0
        self.actor_theta = []
        
        self.H = 500  # Horizon, max number of element within tensor (max time step)
        self.states = np.zeros((self.H, self.state_space))
        self.rewards = np.zeros((self.H, 1))
        self.value = np.zeros((self.H, 1))
        self.actions = np.zeros((self.H, 1))
        self.probs = np.zeros((self.H, self.action_space))
        
        self.gamma = 0.98  # Discount factor
        self.K = 1  # Number of epochs
        self.e = 0.1  # Policy distance
        
        #Make actor and critic
        self.actor, self.critic = self.make_model()
        
        #Optimizer for models
        self.optimizer_actor  = optim.Adam(self.actor.parameters() , lr=0.004)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.08)
        

        #Learning rate update
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=300,gamma=0.9)
        
    
    def make_model(self):
        circ_actor = self.qrlPQC(1, 1024)
        circ_critic = self.qrlPQC(1, 1024)
        
        qnn_actor = EstimatorQNN(
            circuit=circ_actor.circuit,
            input_params=circ_actor.beta.params,
            weight_params=[circ_actor.theta],
            input_gradients=True
        ) 
        qnn_critic = EstimatorQNN(
            circuit=circ_critic.circuit,
            input_params=circ_critic.beta.params,
            weight_params=[circ_critic.theta],
            input_gradients=True
        )
        
        qlayer_actor = TorchConnector(qnn_actor)
        qlayer_critic = TorchConnector(qnn_critic)
        
        customRepeat_actor  = self.ExpectAndRepeatLayer()
        customRepeat_critic = self.ExpectAndRepeatLayer()
        
        #Actor model
        model_actor = torch.nn.Sequential(qlayer_actor, customRepeat_actor, torch.nn.Linear(64, 2), torch.nn.Softmax(dim=-1))
        #Critic model
        model_critic = torch.nn.Sequential(qlayer_critic, customRepeat_critic, torch.nn.Linear(64, 1))
        
        return model_actor, model_critic
        
    class qrlPQC: 
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
            
            self.circuit.rx(self.theta, all_qubits)
            
            
            #self.circuit.measure_all()
            # ---------------------------
    
            self.shots = shots
        
    class ExpectAndRepeatLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.number_of_reuse = 64
        
        def forward(self, x):
            #exp_val = 1*x[0] + (-1)*x[1]
            return x.repeat(self.number_of_reuse)
        
    def convert_data(self, x):
        
        beta0 = math.atan(x[0])
        beta1 = math.atan(x[1])
        beta2 = math.atan(x[2])
        beta3 = math.atan(x[3])
        
        s = [beta0, beta1, beta2, beta3]
        
        return s
    
    def softmax(self, x):
        
        f_0 = np.exp(x[0]) / np.sum(np.exp(x))
        f_1 = np.exp(x[1]) / np.sum(np.exp(x))
        return [f_0,f_1]
    
    #Use the Actor model to pick action, critic to estimate value
    def pick_action(self, obs):

      tensor_obs = torch.Tensor(obs[1:4])     
      
      actor_output =   self.actor(tensor_obs)
      #print(actor_output)
      prob1 = actor_output[0].item()
      prob2 = actor_output[1].item()
      probs =[prob1,prob2]
      probs = self.softmax(probs)
      print(probs)
      action = np.random.choice(2, p = probs  )
      
      critic_output = self.critic(tensor_obs)
      value =  critic_output.item()
  
      return action, probs, value
  
    def remember(self, state, reward, action, probs, vals):
        
        i = self.iter
        self.states[i] = state
        self.rewards[i] = reward
        self.value[i] = vals
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
        
        ratio = cur_pol/old_pol
        ratio = torch.clip(ratio, 1e-10, 10 - 1e-10)
        clipped = torch.clip(ratio, 1 - self.e, 1 + self.e)
        mul1 = advantages * ratio 
        mul2 = advantages *clipped 
        loss = -torch.min(mul1  , mul2 ).mean()

        return loss
    
    def learn(self):
        
        # for t in range(self.K):
        #print('Epoch: '+  str(t))
        state_batch = torch.Tensor(self.states[:self.iter])
        #print(state_batch)
        p_batch = torch.Tensor(self.probs[:self.iter])
        p_batch = p_batch.double()
        print(p_batch)
        v_batch = torch.Tensor(self.value[:self.iter])
        v_batch = v_batch.double()
        #print(v_batch)
        action_batch = torch.zeros((self.iter,1),dtype=torch.int64)
        for i in range(self.iter):              
            action_batch[i,0] = self.actions[i,0]
        action_batch = torch.Tensor(action_batch)
        #print(action_batch)
        rewards = self.discount_reward(self.rewards[:self.iter]) #Calculate discounted reward sum
        rewards = torch.Tensor(rewards)
        #print(rewards)
         
        for t in range(self.K):
            value_tensor = torch.zeros(state_batch.shape[0],1)
            actor_predict = torch.zeros((state_batch.shape[0],2))
 
            #Critic loss
            for i in range(state_batch.shape[0]):
                value_tensor[i,0] = self.critic(state_batch[i,1:4])                                        
            critic_loss = torch.mean((value_tensor - rewards)**2)
            print("critic loss: " +str(critic_loss))
            
            #Actor loss
            #with torch.no_grad():
            advantage = rewards - value_tensor         
            #advantage = rewards - v_batch
            #advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8) #advantages normalized
            advantage = torch.transpose(advantage,0,1)
            for i in range(state_batch.shape[0]):              
                actor_predict[i] = self.actor(state_batch[i,1:4])
            old_actor = torch.gather(p_batch, 0, action_batch)
            new_actor = torch.gather(actor_predict, 0, action_batch)
            actor_loss = self.ppo_loss(new_actor, old_actor, advantage)
            print("actor loss: " +str(actor_loss))
            
            #Optimize
            self.optimizer_critic.zero_grad()         
            self.optimizer_actor.zero_grad()
            critic_loss.backward(retain_graph=True) #gradient descent of critic loss
            actor_loss.backward(retain_graph=True) #gradient descent (ascent??) of actor loss
            
            print(self.actor[0].weight)
            print(self.critic[0].weight)
            self.optimizer_critic.step() #how to check theta??
            self.optimizer_actor.step()
            self.scheduler.step()
            
        self.iter = 0
        print("Actor NN bias " + str(self.actor[2].bias))
        #print("Actor NN weight 1" + str(self.actor[2].weight[0]))
        print("Actor NN weight 1" + str(torch.sum(self.actor[2].weight[0])))
        print("Actor NN weight 2" + str(torch.sum(self.actor[2].weight[1])))
        
        # return model_actor[0].weight.item(), model_critic[0].weight.item()
        return actor_loss.item(), critic_loss.item(), self.actor[0].weight.item()
        
#Load Environment

environment_name = "CartPole-v1"
env = gym.make(environment_name)
episodes = 2000
PPO_agent = QNN_PPO_Agent(env.action_space.n, env.observation_space.shape)
actor_arr =  []
critic_arr = []
score_arr = []            

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
     
for episode in range(1, episodes+1):
    ini_state = env.reset()    
    done = False
    score = 0 
    
    # Bloch sphere plot formatting   
    b1 = Bloch()
    b2 = Bloch()
    b1.point_color = ['tab:blue']
    b2.point_color = ['tab:blue']
    b1.point_marker = ['o']
    b2.point_marker = ['o']
    b1.point_size=[2]
    b2.point_size=[2]
    
    while not done:
        #env.render()       
        
        obs1.append(math.atan(ini_state[1]))
        obs2.append(math.atan(ini_state[2]))
        obs3.append(math.atan(ini_state[3]))
        
        converted_state = PPO_agent.convert_data(ini_state)
        action, prob, vals = PPO_agent.pick_action(converted_state)
        n_state, reward, done, info = env.step(action)
        score+=reward
        PPO_agent.remember(ini_state, reward, action, prob, vals)
        ini_state = n_state
        
    #Show states on Bloch sphere    
    # for e in range(len(obs1)):  
    #     state_1=Statevector.from_instruction(qc_A.bind_parameters({beta1_param:obs1[e], beta2_param:obs2[e], beta3_param:obs3[e]}))          
    #     b1.add_points(state_to_bloch(state_1))            
    # b1.show()
    
    actor, critic, param = PPO_agent.learn()
    actor_arr.append(actor) 
    critic_arr.append(critic)
    score_arr.append(score)
    
    #Show states with parameter theta
    for n in range(len(obs1)):        
      state_2=Statevector.from_instruction(qc_B.bind_parameters({beta1_param:obs1[n], beta2_param:obs2[n], beta3_param:obs3[n], theta_param:param})) 
      b2.add_points(state_to_bloch(state_2))
    b2.show()
    
    obs1 = []
    obs2 = []
    obs3 = []
    
    print('Episode:{} Score:{}'.format(episode, score))
env.close()