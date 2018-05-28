import gym
import numpy as np
from gym import spaces
import math

"""
The Puddleworld environment

Info
----
  - State space: 2D Box (x,y)
  - Action space: Discrete (UP,DOWN,RIGHT,LEFT)
  - Parameters: goal position x and y, puddle centers, puddle variances

References
----------
  
  - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
    Importance Weighted Transfer of Samples in Reinforcement Learning.
    International Conference on Machine Learning. 2018.
    
  - https://github.com/amarack/python-rl/blob/master/pyrl/environments/puddleworld.py
"""

class PuddleWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, goal_x=10, goal_y=10, puddle_means=[(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], 
                 puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)],
                 puddle_slow = True):

        self.horizon = 50
        self.gamma = 0.99
        self.state_dim = 2
        self.action_dim = 1
        
        self.size = np.array([10,10])
        self.goal = np.array([goal_x,goal_y])
        self.noise = 0.2
        self.reward_noise = 0.1
        self.fudge = 1.41
        self.puddle_slow = 5 if puddle_slow else 0

        self.puddle_penalty = -100.0
        self.puddle_means = list(map(np.array, puddle_means))

        self.puddle_var = list(map(lambda cov: np.linalg.inv(np.array(cov).reshape((2,2))), puddle_var))
        self.puddles = list(zip(self.puddle_means, self.puddle_var))

        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([10,10]))

        self.action_space = spaces.Discrete(4)

        self.seed()
        self.reset()

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            self.pos = np.array([0., 0.])
        else:
            self.pos = np.array(state)
        return self.get_state()

    def isAtGoal(self, pos):
        return np.linalg.norm(pos - self.goal) < self.fudge

    def mvnpdf(self, x, mu, sigma_inv):
        size = len(x)
        if size == len(mu) and sigma_inv.shape == (size, size):
            det = 1.0 / np.linalg.det(sigma_inv)
            norm_const = 1.0 / ( math.pow((2*np.pi),float(size)/2) * math.pow(det,0.5) )
            x_mu = x - mu
            result = math.pow(math.e, -0.5 * np.dot(x_mu, np.dot(sigma_inv, x_mu)))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")
        
    def _get_puddle_weight(self, pos):
        
        weight = 0
        for mu, inv_cov in self.puddles:
            weight += self.mvnpdf(pos, mu, inv_cov)
        return weight

    def step(self, a):
        
        s = self.get_state()

        puddle_weight = self._get_puddle_weight(self.pos)

        alpha = 1 / (1 + (self.puddle_slow * puddle_weight))

        if int(a) == 0:
            self.pos[0] += alpha
        elif int(a) == 1:
            self.pos[0] -= alpha
        elif int(a) == 2:
            self.pos[1] += alpha
        elif int(a) == 3:
            self.pos[1] -= alpha

        if self.noise > 0:
            self.pos += np.random.normal(scale=self.noise, size=(2,))
        self.pos = self.pos.clip([0,0],self.size)

        base_reward = 0.0 if self.isAtGoal(s) else -1.0
        base_reward += puddle_weight * self.puddle_penalty

        if self.reward_noise > 0:
            base_reward += np.random.normal(scale=self.reward_noise)

        if self.isAtGoal(self.pos):
            self._absorbing = True
        else:
            self._absorbing = False
        
        return self.get_state(), base_reward, self._absorbing, {}

    def get_state(self):
        return np.array(self.pos)
    
    def get_reward_mean(self, s, a):
        
        puddle_weight = self._get_puddle_weight(s)
        reward = 0.0 if self.isAtGoal(s) else -1.0
        reward += puddle_weight * self.puddle_penalty
        return reward

    def get_transition_mean(self, s, a):
        
        puddle_weight = self._get_puddle_weight(s)
        alpha = 1 / (1 + (self.puddle_slow * puddle_weight))
        next_s = np.array(s)
        
        if int(a) == 0:
            next_s[0] += alpha
        elif int(a) == 1:
            next_s[0] -= alpha
        elif int(a) == 2:
            next_s[1] += alpha
        elif int(a) == 3:
            next_s[1] -= alpha
        
        return next_s.clip([0,0],self.size)
    