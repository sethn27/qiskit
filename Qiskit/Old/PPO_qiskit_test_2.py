import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

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

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits, obs, backend, shots):
        # --- Convert Data ---
        beta0 = math.atan(obs[0])
        beta1 = math.atan(obs[1])
        beta2 = math.atan(obs[2])
        beta3 = math.atan(obs[3])
        
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        
        self._circuit.rz(beta1, all_qubits)
        self._circuit.ry(beta2, all_qubits)
        self._circuit.rz(beta3, all_qubits)
        self._circuit.barrier()
        
        self._circuit.rz(self.theta, all_qubits)
        
        
        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        t_qc = transpile(self._circuit,
                         self.backend)
        qobj = assemble(t_qc,
                        shots=self.shots,
                        parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])


#Load Environment

environment_name = "CartPole-v0"
env = gym.make(environment_name)
episodes = 5
simulator = qiskit.Aer.get_backend('aer_simulator')


for episode in range(1, episodes+1):
    state = env.reset()  
    done = False
    score = 0 
    
    circuit = QuantumCircuit(1, state, simulator, 100)
    print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
    print(circuit._circuit.draw(output='mpl'))
    print(circuit._circuit)
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))

