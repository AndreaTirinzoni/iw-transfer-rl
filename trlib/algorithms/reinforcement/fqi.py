import numpy as np
from trlib.algorithms.algorithm import Algorithm
from copy import deepcopy
from gym import spaces
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes
from trlib.utilities.evaluation import evaluate_policy

class FQI(Algorithm):
    """
    Fitted Q-Iteration
    
    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """
    
    def __init__(self, mdp, policy, n_episodes, actions, batch_size, max_iterations, regressor_type, verbose = False, **kwargs):
        
        super().__init__(mdp, policy, n_episodes, verbose)
        
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        
        regressor = regressor_type(**kwargs)
        
        if isinstance(mdp.action_space, spaces.Discrete):
            self.Q = DiscreteFittedQ([deepcopy(regressor) for _ in actions], mdp.state_dim)
        else:
            self.Q = FittedQ(regressor, mdp.state_dim, mdp.action_dim)
            
        self.policy.Q = self.Q
        
        self._steps = 0
        self._episodes = 0
        self._iteration = 0
        self._data = []
        
        self._a_idx = 1 + mdp.state_dim
        self._r_idx = self._a_idx + mdp.action_dim
        self._s_idx = self._r_idx + 1
        
    def _iter(self, sa, r, s_prime, absorbing, **kwargs):

        self.display("Iteration {0}".format(self._iteration))
        
        if self._iteration == 0:
            y = r
        else:
            maxq, _ = self.Q.max(s_prime, self._actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self.Q.fit(sa, y.ravel(), **kwargs)

        self._iteration += 1
        
    def step(self, **kwargs):
        
        self.display("Step {0}".format(self._steps))
        
        policy = self.policy if self._steps > 0 else Uniform(self._actions)
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self._episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0
        
        for _ in range(self._max_iterations):
            self._iter(data[:,1:self._r_idx], data[:,self._r_idx:self._s_idx], data[:,self._s_idx:-1], data[:,-1], **kwargs)
            
        perf = evaluate_policy(self._mdp, self.policy, criterion = 'discounted', n_episodes = 4, initial_states = None, n_threads = 1)
        
        return (self._episodes, perf[0])
        
        

    
    
        
    
    