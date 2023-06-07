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
    
    # def run(self, thetas):
    #     t_qc = transpile(self.circuit,
    #                      self.backend)
    #     qobj = assemble(t_qc,
    #                     shots=self.shots,
    #                     parameter_binds = [{self.theta: theta} for theta in thetas])
    #     job = self.backend.run(qobj)
    #     result = job.result().get_counts()
        
    #     counts = np.array(list(result.values()))
    #     states = np.array(list(result.keys())).astype(float)
        
    #     # Compute probabilities for each state
    #     probabilities = counts / self.shots
    #     # Get state expectation
    #     expectation = np.sum(states * probabilities)
        
    #     return np.array([expectation])

class ExpectAndRepeatLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.number_of_reuse = 64
    
    def forward(self, x):
        exp_val = 0*x[0] + 1*x[1]
        return exp_val.repeat(self.number_of_reuse)

#Load Environment

environment_name = "CartPole-v0"
env = gym.make(environment_name)
episodes = 5
circ = qrlPQC(1, 100)
qnn = SamplerQNN(
    circuit=circ.circuit,
    input_params=circ.beta.params,
    weight_params=[circ.theta],
    input_gradients=True
)

qlayer = TorchConnector(qnn)
    
customRepeat = ExpectAndRepeatLayer()

model = torch.nn.Sequential(qlayer, customRepeat, torch.nn.Linear(64, 2), torch.nn.Softmax(dim=0))

test_obs = torch.Tensor([1, 1, 1])
print("Input: " + str(test_obs))
with torch.no_grad():
    output = model(test_obs)
    print("Output: " + str(output))
    print("Action: " + str(torch.argmax(output)))

# for episode in range(1, episodes+1):
#     state = env.reset()
#     output = model(state)
#     # print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
#     # print(circuit.circuit.draw(output='mpl'))
#     # print(circuit.circuit)
    
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
env.close()
