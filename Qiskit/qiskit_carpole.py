
# Useful additional packages
import gym
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from math import pi, sqrt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, assemble, Aer
from qiskit.tools.visualization import circuit_drawer, plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit import *
from numpy import linalg as la
from qiskit.tools.monitor import job_monitor
import qiskit.tools.jupyter

from qiskit import IBMQ
# IBMQ.save_account(TOKEN)
#IBMQ.enable_account('94d92395742a4f2ef1ec1455e8d278ee072c576772e70ac4eef2581a78a7c02aae0b52c4c56cddfa5705c9e69ce794beac9843e9d5617b81a5f7038b2feb5e72')

#provider=IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='quantum-ml')
#provider=IBMQ.get_provider(hub='ibm-q-hub-ntu', group='ntu-internal', project='default')
#backend = provider.get_backend('ibmq_armonk')
backend = Aer.get_backend('aer_simulator_statevector')

env_name = 'CartPole-v1'
import math
EPISODES = 5

# Transfer the state to the angle.
def get_angles(x):
    beta0 = math.atan(x[0])
    beta1 = math.atan(x[1])
    beta2 = math.atan(x[2])
    beta3 = math.atan(x[3])
    return [beta0, beta1, beta2, beta3]

class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction  with the quantum circuit 
    """
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        
        self.theta_0 = qiskit.circuit.Parameter('s1')
        self.theta_1 = qiskit.circuit.Parameter('s2')
        self.theta_2 = qiskit.circuit.Parameter('s3')
        self.theta_3 = qiskit.circuit.Parameter('alpha')

        self._circuit.h(all_qubits)
        self._circuit.rz(self.theta_0, all_qubits)
        self._circuit.ry(self.theta_1, all_qubits)
        self._circuit.rz(self.theta_2, all_qubits)
        self._circuit.rx(self.theta_3, all_qubits)
        
        self._circuit.measure_all()
        # ---------------------------
        self.backend = backend
        self.shots = shots
    
    def run(self, thetas):
        
        s1 = thetas[1]
        s2 = thetas[2]
        s3 = thetas[3]
        # trainable parameters alpha
        alpha = -0.6225512
        
        t_qc = transpile(self._circuit,self.backend)
        qobj = assemble(t_qc,shots=self.shots,
        parameter_binds = [{self.theta_0:s1, self.theta_1:s2, self.theta_2: s3,self.theta_3: alpha}])  
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        #print('Output',states)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        print('Probabilities of 0 and 1',probabilities )
        # Get state expectation
        expectation = np.sum(states * probabilities)
        print('Mark state 1',expectation)        
        expectation = (1-expectation) -expectation 
        print('The diffence between state 1 and 0 (expectation)',expectation)
        return expectation ,expectation

def softmax(x):
    
    f_0 = np.exp(x[0]) / np.sum(np.exp(x))
    f_1 = np.exp(x[1]) / np.sum(np.exp(x))
    return [f_0,f_1]

class VQC():
    def get_action(self, state):
        #print('States: ',state)
        circuit = QuantumCircuit(1, backend, 1024)
        measurement_0,measurement_1 = circuit.run(state)                  
        print('Output Measurement (expectation)',measurement_0,measurement_1)

        # trainable parameters of NN and bias
        measurement_0 = 39.853912 * measurement_0  +0.05803982     
        measurement_1 = -39.853912 * measurement_1 -0.05803982
        #print('Scaled Measurement',measurement_0,measurement_1)

        action_prob = [measurement_0,measurement_1]
        action_prob = softmax(action_prob)
        print('Action Probabilities after Softmax',action_prob)

        action = np.random.choice(2,p=action_prob)
        print('Chosen Action \n',action)
        return action

def main():
    env = gym.make(env_name)
    #env.seed(34)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    get_action=VQC()
    all_rewards = []
    all_episode_action = []
    
    for e in range(EPISODES):
        done = False
        rewards=0
        #change in gym v26
        s1 = env.reset()
        env.render()
        episode_action = []
        while not done:
            state_angle = get_angles(s1)
            action = get_action.get_action(state_angle)
            episode_action.append(action)
            s2, reward, done, _ = env.step(action)[0:4] #change in new version
            
            s1 = s2 
            rewards += reward
            print(done, rewards)

            if rewards >= 500:
                break

        all_rewards.append(rewards)
        print("\rEpisode {}/{} ||  Current Iteration Reward {}".format(e,EPISODES,rewards))
        print(episode_action)
        all_episode_action.append(episode_action)
    print(all_episode_action)

    #np.save("VQC Qiskit ppo testing circuit V0.0.0 HRzRyRz Rx measurement 1 1qubits rewards", np.asarray(all_rewards))
    plt.title("VQC Qiskit ppo testing circuit V0.0.1 HRzRyRz Rx measurement 1 1qubits ")
    plt.plot(all_rewards, label='Reward')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.show()
    env.close()


if __name__=="__main__":
    main()

