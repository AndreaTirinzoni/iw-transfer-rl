import numpy as np
from trlib.algorithms.algorithm import Algorithm
from gym import spaces
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes

class FQI(Algorithm):
    """
    Fitted Q-Iteration
    
    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, verbose = False, **regressor_params):
        
        super().__init__("FQI", mdp, policy, verbose)
        
        self._actions = actions
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._regressor_type = regressor_type
        
        if isinstance(mdp.action_space, spaces.Discrete):
            self._policy.Q = DiscreteFittedQ(regressor_type, mdp.state_dim, len(actions), **regressor_params)
        else:
            self._policy.Q = FittedQ(regressor_type, mdp.state_dim, mdp.action_dim, **regressor_params)
        
        self._a_idx = 1 + mdp.state_dim
        self._r_idx = self._a_idx + mdp.action_dim
        self._s_idx = self._r_idx + 1
        
        self.reset()
        
    def _iter(self, sa, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))
        
        if self._iteration == 0:
            y = r
        else:
            maxq, _ = self._policy.Q.max(s_prime, self._actions, absorbing)
            y = r.ravel() + self._mdp.gamma * maxq

        self._policy.Q.fit(sa, y.ravel(), **fit_params)

        self._iteration += 1
        
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 0 else Uniform(self._actions)
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0
        
        for _ in range(self._max_iterations):
            self._iter(data[:,1:self._r_idx], data[:,self._r_idx:self._s_idx], data[:,self._s_idx:-1], data[:,-1], **kwargs)
            
        self._result.update_step(n_episodes = self.n_episodes, n_samples = data.shape[0])
    
    def reset(self):
        
        super().reset()
        
        self._data = []
        self._iteration = 0
        
        self._result.add_fields(batch_size=self._batch_size, max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__), policy = str(self._policy.__class__.__name__))
    