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

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
import qiskit_machine_learning
from qiskit_machine_learning.neural_networks import SamplerQNN,CircuitQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN
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
        
        self.circuit.rx(self.theta, all_qubits)
        
        
        #self.circuit.measure_all()
        # ---------------------------

        self.shots = shots

class qrlPQC2:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, shots):
        
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        
        self.alpha = qiskit.circuit.Parameter('alpha')
        self.beta = qiskit.circuit.ParameterVector('beta', 3)
        
        
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        
        self.circuit.rz(self.beta[0], all_qubits)
        self.circuit.ry(self.beta[1], all_qubits)
        self.circuit.rz(self.beta[2], all_qubits)
        self.circuit.barrier()
        
        self.circuit.rx(self.alpha, all_qubits)
        
        
        #self.circuit.measure_all()
        # ---------------------------

        self.shots = shots
    


class ExpectAndRepeatLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.number_of_reuse = 32
    
    def forward(self, x):
        #exp_val = 1*x[0] + (-1)*x[1]
        return x.repeat(self.number_of_reuse)

#is expectation correct for critic??
class ExpectAndRepeatLayer2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.number_of_reuse = 32
    
    def forward(self, x):
        #exp_val = 1*x[0] + (-1)*x[1]
        return x.repeat(self.number_of_reuse)

    
    
#Define actor qnn model:
circ = qrlPQC(1, 1024)
qnn = EstimatorQNN(
    circuit=circ.circuit,
    input_params=circ.beta.params,
    weight_params=[circ.theta],
    input_gradients=True
)

qlayer = TorchConnector(qnn)
    
customRepeat = ExpectAndRepeatLayer()

#Actor model
model_actor = torch.nn.Sequential(qlayer, customRepeat, torch.nn.Linear(32, 2), torch.nn.Softmax(dim=-1))
total_params = sum(p.numel() for p in model_actor.parameters())
print(f"Number of parameters: {total_params}")



#Define critic qnn model:
circ_2 = qrlPQC2(1, 1024)
qnn_2 = EstimatorQNN(
    circuit=circ_2.circuit,
    input_params=circ_2.beta.params,
    weight_params=[circ_2.alpha],
    input_gradients=True
)

qlayer_2 = TorchConnector(qnn_2)
    
customRepeat_2 = ExpectAndRepeatLayer2()

#Critic model
model_critic = torch.nn.Sequential(qlayer_2, customRepeat_2, torch.nn.Linear(32,1))


#Optimizer for parameter
optimizer_critic = optim.Adam(model_critic.parameters(), lr=0.04)
optimizer_actor  = optim.Adam(model_actor.parameters() , lr=0.004)

def softmax(x):
    
    f_0 = np.exp(x[0]) / np.sum(np.exp(x))
    f_1 = np.exp(x[1]) / np.sum(np.exp(x))
    return [f_0,f_1]

def convert_data(x):
    
    beta0 = math.atan(x[0])
    beta1 = math.atan(x[1])
    beta2 = math.atan(x[2])
    beta3 = math.atan(x[3])
    
    s = [beta0, beta1, beta2, beta3]
    
    return s


#Use the Actor model to pick action, critic to estimate value
def pick_action(obs):
    
  tensor_obs = torch.Tensor(obs[1:4])     
  
  actor_output =   model_actor(tensor_obs)
  #print(actor_output)
  prob1 = actor_output[0].item()
  prob2 = actor_output[1].item()
  probs =[prob1,prob2]
  probs = softmax(probs)
  # print(probs)
  action = np.random.choice(2, p = probs  )
  
  # action = torch.argmax(actor_output)
  # action = action.item()
  
  critic_output = model_critic(tensor_obs)
  value =  critic_output.item()
  #print(value)
  
   
  return action, probs, value

#logprob = dist.log_prob(sample) means to get the logarithmic probability (logprob) 
#of one experiment sample (sample) under a specific distribution (dist).

a = pick_action([1, 2, 3, 4])
class Agent(object):
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
        #print("ratio: " +str(ratio))
        ratio = torch.clip(ratio, 1e-10, 10 - 1e-10)
        clipped = torch.clip(ratio, 1 - self.e, 1 + self.e)
        #print(clipped)
        # mul1 = torch.mul(advantages , ratio)
        # mul2 = torch.mul(advantages , clipped)
        # print(mul1, mul2)
        mul1 = advantages * ratio 
        mul2 = advantages *clipped 
        #print(mul1, mul2)
        loss = -torch.min(mul1  , mul2 ).mean()

        return loss
    
    def log_prob(self,probs, action):
        dist = Categorical(probs)
        #action = dist.sample()
        log_prob_out =dist.log_prob(action)
        return log_prob_out
    
    def learn(self):
        # batch_indices = [i for i in range(self.iter)]
        # print("a" +str(batch_indices))
        for t in range(self.K):
            #print('Epoch: '+  str(t))
            state_batch = torch.Tensor(self.states[:self.iter])
            #print(state_batch)
            
            p_batch = torch.Tensor(self.probs[:self.iter])
            p_batch = p_batch.double()
            print(p_batch)
            
            v_batch = torch.Tensor(self.value[:self.iter])
            #v_batch = v_batch.double()
            #print(v_batch)
            
            action_batch = torch.zeros((self.iter,1),dtype=torch.int64)
            for i in range(self.iter):              
                action_batch[i,0] = self.actions[i,0]
            action_batch = torch.Tensor(action_batch)
            # action_batch = torch.int64(action_batch)
            #print(action_batch)

            rewards = self.discount_reward(self.rewards[:self.iter]) #Calculate discounted reward sum
            rewards = torch.Tensor(rewards)
            
            #print(rewards)
         
            value_tensor = torch.zeros(state_batch.shape[0],1)
            actor_predict = torch.zeros((state_batch.shape[0],2))
        
        
            for i in range(state_batch.shape[0]):
                value_tensor[i,0] = model_critic(state_batch[i,1:4])                                      
                
            
            #print("value_tensor:" +str( value_tensor))
            #critic_loss = torch.mean((value_tensor - rewards)**2)
            critic_loss = torch.mean((rewards - value_tensor )**2)
            #print("critc loss: " +str(critic_loss))
            
            with torch.no_grad():
                advantage = rewards - value_tensor 
            #advantage = rewards + v_batch              
            #advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8) #advantages normalized
            advantage = torch.transpose(advantage,0,1)
            #print("advantage: " +str(advantage))
            
            
            
            
            for i in range(state_batch.shape[0]):              
                actor_predict[i] = model_actor(state_batch[i,1:4])
            #print(actor_predict)  
            #print(p_batch)
            
            
            old_actor = torch.gather(p_batch, 0, action_batch)
            # old_actor = self.log_prob(p_batch, action_batch)
            #print(old_actor)
            new_actor = torch.gather(actor_predict, 0, action_batch)
            # new_actor = self.log_prob(actor_predict,action_batch)
            #print(new_actor)
            
 
            rewards = torch.transpose(rewards,0,1)
            actor_loss = self.ppo_loss(new_actor, old_actor, rewards)
            print("actor loss: " +str(actor_loss))
            
            #total_loss = actor_loss + 0.7*critic_loss
            
            optimizer_critic.zero_grad()         
            optimizer_actor.zero_grad()
            
            actor_loss.backward() #gradient descent (ascent??) of actor loss
            critic_loss.backward() #gradient descent of critic loss
            
            #total_loss.backward()
            print(model_actor[0].weight)
            print(model_critic[0].weight)
            
            optimizer_critic.step() #how to check theta??
            optimizer_actor.step()
     
        self.iter = 0
        # return model_actor[0].weight.item(), model_critic[0].weight.item()
        return actor_loss.item(), critic_loss.item()
#Load Environment

environment_name = "CartPole-v1"
env = gym.make(environment_name)
episodes = 1000
PPO_agent = Agent(env.action_space.n, env.observation_space.shape)
actor_arr =  []
critic_arr = []
score_arr = []


for episode in range(1, episodes+1):
    ini_state = env.reset()    
    done = False
    score = 0 
    
    while not done:
        #env.render()       
        
        converted_state = convert_data(ini_state)
        action, prob, vals = pick_action(converted_state)
        
        #print(action)
        n_state, reward, done, info = env.step(action)
        score+=reward
        PPO_agent.remember(converted_state, reward, action, prob, vals)
        
        ini_state = n_state
    actor, critic = PPO_agent.learn()
    actor_arr.append(actor) 
    critic_arr.append(critic)
    score_arr.append(score)
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

