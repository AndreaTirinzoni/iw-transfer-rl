import numpy as np
from trlib.algorithms.algorithm import Algorithm
from copy import deepcopy
from gym import spaces
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes
from trlib.utilities.evaluation import evaluate_policy
from trlib.experiments.results import Result

class FQI(Algorithm):
    """
    Fitted Q-Iteration
    
    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """
    
    def __init__(self, mdp, policy, max_steps, actions, batch_size, max_iterations, regressor_type, eval_policy = True, eval_episodes = 1, eval_threads = 1, verbose = False, **kwargs):
        
        super().__init__(mdp, policy, max_steps, verbose)
        
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._eval_policy = eval_policy
        self._eval_episodes = eval_episodes
        self._eval_threads = eval_threads
        self._regressor_type = regressor_type
        
        regressor = regressor_type(**kwargs)
        
        if isinstance(mdp.action_space, spaces.Discrete):
            self.policy.Q = DiscreteFittedQ([deepcopy(regressor) for _ in actions], mdp.state_dim)
        else:
            self.policy.Q = FittedQ(regressor, mdp.state_dim, mdp.action_dim)
        
        self._a_idx = 1 + mdp.state_dim
        self._r_idx = self._a_idx + mdp.action_dim
        self._s_idx = self._r_idx + 1
        
        self.reset()
        
    def _iter(self, sa, r, s_prime, absorbing, **kwargs):

        self.display("Iteration {0}".format(self._iteration))
        
        if self._iteration == 0:
            y = r
        else:
            maxq, _ = self.policy.Q.max(s_prime, self._actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self.policy.Q.fit(sa, y.ravel(), **kwargs)

        self._iteration += 1
        
    def step(self, callbacks = [], **kwargs):
        
        self.display("Step {0}".format(self._step))
        
        policy = self.policy if self._step > 0 else Uniform(self._actions)
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size, self._eval_threads))
        self._episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0
        
        for _ in range(self._max_iterations):
            self._iter(data[:,1:self._r_idx], data[:,self._r_idx:self._s_idx], data[:,self._s_idx:-1], data[:,-1], **kwargs)
        
        
        perf = evaluate_policy(self._mdp, self.policy, criterion = 'discounted', n_episodes = self._eval_episodes, initial_states = None, n_threads = self._eval_threads) if self._eval_policy else 0
        
        self.result.add_step(step=self._step,n_episodes=self._episodes,perf=perf)
        
        for cb in callbacks:
            cb(self)
        
        self._step += 1
        
        return self.result
        
    def run(self, callbacks = [], **kwargs):
        
        for _ in range(self._max_steps):
            self.step(callbacks, **kwargs)

        return self.result
    
    def reset(self):
        
        self._step = 0
        self._iteration = 0
        self._data = []
        self._episodes = 0
        self.result = Result("FQI", batch_size=self._batch_size, max_iterations=self._max_iterations, regressor_type=str(self._regressor_type.__name__), policy = str(self.policy.__class__.__name__))
    
    