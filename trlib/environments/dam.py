import gym
import numpy as np
from gym import spaces

"""
Cyclostationary Dam Control

Info
----
  - State space: 2D Box (storage,day)
  - Action space: 1D Box (release decision)
  - Parameters: capacity, demand, flooding threshold, inflow mean per day, inflow std, demand weight, flooding weigt=ht

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)
    
  - A. Castelletti, S. Galelli, M. Restelli, R. Soncini-Sessa
    Tree-based reinforcement learning for optimal water reservoir operation
    Water Resources Research 46.9 (2010)
"""

class Dam(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.5, beta = 0.5, penalty_on = False):
        
        self.horizon = 360
        self.gamma = 0.999
        self.state_dim = 2
        self.action_dim = 1

        self.DEMAND = demand  # Water demand -> At least DEMAND/day must be supplied or a cost is incurred
        self.FLOODING = flooding  # Flooding threshold -> No more than FLOODING can be stored or a cost is incurred
        self.CAPACITY = capacity  # Maximum storage capacity -> At least max{S - CAPACITY, 0} must be released
        self.INFLOW_MEAN = self._get_inflow_profile(inflow_profile)  # Random inflow (e.g. rain) mean for each day (360-dimensional vector)
        self.INFLOW_STD = inflow_std # Random inflow std
        
        assert alpha + beta == 1.0 # Check correctness
        self.ALPHA = alpha # Weight for the flooding cost
        self.BETA = beta # Weight for the demand cost
        
        self.penalty_on = penalty_on # Whether to penalize illegal actions or not
        
        # Gym attributes
        self.viewer = None
        
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([30]))  
        
        self.observation_space = spaces.Box(low=np.array([0,1]),
                                            high=np.array([np.inf,360]))

        # Initialization
        self.seed()
        self.reset()
    
    def _get_inflow_profile(self,n):
        
        assert n >= 1 and n <=4
        
        if n == 1:
            return self._get_inflow_1()
        elif n == 2:
            return self._get_inflow_2()
        elif n == 3:
            return self._get_inflow_3()
        else:
            return self._get_inflow_4()
    
    def _get_inflow_1(self):
        
        y = np.zeros(360)  
        for x in range(360):
            if x < 120:
                y[x] = np.sin(x * 3 * np.pi / 359) + 0.5
            elif x < 240:
                y[x] = np.sin(x * 3 * np.pi / 359) / 2 + 0.5
            else:
                y[x] = np.sin(x * 3 * np.pi / 359) + 0.5
        return y * 8 + 2
    
    def _get_inflow_2(self):
        
        y = np.zeros(360)  
        for x in range(360):
            if x < 120:
                y[x] = np.sin(x * 3 * np.pi / 359) / 2 + 0.25
            elif x < 240:
                y[x] = np.sin(x * 3 * np.pi / 359 + np.pi) * 3 + 0.25
            else:
                y[x] = np.sin(x * 3 * np.pi / 359 + np.pi) / 4 + 0.25
        return y * 8 + 2
    
    def _get_inflow_3(self):
        
        y = np.zeros(360)  
        for x in range(360):
            if x < 120:
                y[x] = np.sin(x * 3 * np.pi / 359) * 3 + 0.25
            elif x < 240:
                y[x] = np.sin(x * 3 * np.pi / 359) / 4 + 0.25
            else:
                y[x] = np.sin(x * 3 * np.pi / 359) / 2 + 0.25
        return y * 8 + 2
    
    def _get_inflow_4(self):
        
        y = np.zeros(360)  
        for x in range(360):
            if x < 120:
                y[x] = np.sin(x * 3 * np.pi / 359) + 0.5
            elif x < 240:
                y[x] = np.sin(x * 3 * np.pi / 359) / 2.5 + 0.5
            else:
                y[x] = np.sin(x * 3 * np.pi / 359) + 0.5
        return y * 7 + 2
        
    def step(self, action):
        
        # Get current state
        state = self.get_state()
        storage = state[0]
        day = state[1]
        
        # Bound the action
        actionLB = max(storage - self.CAPACITY, 0.0)
        actionUB = storage

        # Penalty proportional to the violation
        bounded_action = min(max(action, actionLB), actionUB)
        penalty = -abs(bounded_action - action) * self.penalty_on

        # Transition dynamics
        action = bounded_action
        inflow = self.INFLOW_MEAN[int(day-1)] + np.random.randn() * self.INFLOW_STD
        nextstorage = max(storage + inflow - action, 0.0)

        # Cost due to the excess level wrt the flooding threshold
        reward_flooding = -max(storage - self.FLOODING, 0.0) / 6 + penalty

        # Deficit in the water supply wrt the water demand
        reward_demand = -max(self.DEMAND - action, 0.0) ** 2 + penalty
        
        # The final reward is a weighted average of the two costs
        reward = self.ALPHA * reward_flooding + self.BETA * reward_demand

        # Get next day
        nextday = day + 1 if day < 360 else 1

        self.state = [nextstorage, nextday]

        return self.get_state(), reward, False, {}

    def reset(self, state=None):
        
        if state is None:
            init_days = np.array([1, 120, 240])
            self.state = [np.random.uniform(0.0, self.CAPACITY), init_days[np.random.randint(low=0,high=3)]]
        else:
            self.state = state

        return self.get_state()

    def get_state(self):
        return np.array(self.state)
    
    def get_reward(self,storage,action):
        
        # Bound the action
        actionLB = max(storage - self.CAPACITY, 0.0)
        actionUB = storage

        # Penalty proportional to the violation
        bounded_action = min(max(action, actionLB), actionUB)
        penalty = -abs(bounded_action - action) * self.penalty_on

        # Transition dynamics
        action = bounded_action

        # Cost due to the excess level wrt the flooding threshold
        reward_flooding = -max(storage - self.FLOODING, 0.0) / 6 + penalty

        # Deficit in the water supply wrt the water demand
        reward_demand = -max(self.DEMAND - action, 0.0) ** 2 + penalty
        
        # The final reward is a weighted average of the two costs
        reward = self.ALPHA * reward_flooding + self.BETA * reward_demand
        
        return reward