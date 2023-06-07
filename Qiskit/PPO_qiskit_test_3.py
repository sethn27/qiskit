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
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

beta = qiskit.circuit.ParameterVector('beta',length=4)
theta = qiskit.circuit.Parameter('theta')

qc1 = qiskit.QuantumCircuit(1)

qc1.h(0)

qc1.rz(beta[1],0)
qc1.ry(beta[2],0)
qc1.rz(beta[3],0)

qc1.rx(theta,0)

# defining the circuit

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        
        all_qubits = [i for i in range(n_qubits)]
        self.beta = qiskit.circuit.ParameterVector('beta', length=4)
        self.theta = qiskit.circuit.Parameter('theta')
        
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        
        self._circuit.rz(beta[1], all_qubits)
        self._circuit.ry(beta[2], all_qubits)
        self._circuit.rz(beta[3], all_qubits)
        self._circuit.barrier()
        
        self._circuit.rz(self.theta, all_qubits)
        
        
        self._circuit.measure_all()
        # ---------------------------

circ = QuantumCircuit(1)

qnn = SamplerQNN(
    circ,
    input_params = circ.beta,
    weight_params = circ.theta
)
    
quantumlayer = TorchConnector(qnn) # torch.nn.Module object

# defining the net

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(8, 2)
      self.fc2 = nn.Linear(2, 8)
      self.qlayer = quantumlayer

    # x represents our data
    def forward(self, x):
      x = self.qlayer(x) # statistics are output

    #   x = torch.repeat(4) # gradients??
      x = self.fc2(x)
      
      x = nn.Softmax(4)

      x = self.fc1(x)

      output = nn.Softmax(x)
      # Apply softmax to x
      return output





