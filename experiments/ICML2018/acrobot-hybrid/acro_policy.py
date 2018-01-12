from trlib.policies.policy import Policy
import numpy as np

class AcrobotPolicy(Policy):
    
    def __init__(self, epsilon = 0):
        self._epsilon = epsilon

    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return np.random.randint(2)
        
        if state[0] > 0 and state[2] > 0:
            return 0
        elif state[0] > 0 and state[2] <= 0:
            return 1
        elif state[0] <= 0 and state[2] < 0:
            return 1
        else:
            return 0