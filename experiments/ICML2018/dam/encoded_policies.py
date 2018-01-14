from trlib.policies.policy import Policy
import numpy as np

class DamPolicy(Policy):

    def __init__(self, epsilon = 0):
        self._epsilon = epsilon
        self._actions = [0, 3, 5, 7, 10, 15, 20, 30]
        
class DamPolicyT(DamPolicy):
    
    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return self._actions[np.random.randint(8)]
        
        day = state[1]
        
        if day < 60:
            return 15.0
        elif day < 120:
            return 10.0
        elif day < 180:
            return 7.0
        elif day < 240:
            return 7.0
        elif day < 300:
            return 10.0
        else:
            return 15.0
        
class DamPolicyS1(DamPolicy):
    
    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return self._actions[np.random.randint(8)]
        
        day = state[1]
        
        if day < 60:
            return 10
        elif day < 120:
            return 10
        elif day < 180:
            return 30
        elif day < 240:
            return 20
        elif day < 300:
            return 7
        else:
            return 5
        
class DamPolicyS2(DamPolicy):
    
    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return self._actions[np.random.randint(8)]
        
        day = state[1]
        
        if day < 60:
            return 20
        elif day < 120:
            return 20
        elif day < 180:
            return 7
        elif day < 240:
            return 7
        elif day < 300:
            return 7
        else:
            return 10
        
class DamPolicyS3(DamPolicy):
    
    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return self._actions[np.random.randint(8)]
        
        day = state[1]
        
        if day < 60:
            return 15
        elif day < 120:
            return 10
        elif day < 180:
            return 5
        elif day < 240:
            return 7
        elif day < 300:
            return 10
        else:
            return 15