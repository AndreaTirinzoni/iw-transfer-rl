import numpy as np
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.utilities.interaction import generate_episodes
from trlib.policies.policy import Uniform

class Laroche2017(FQI):
    """
    Transfer Reinforcement Learning with Shared Dynamics
    
    References
    ----------
    
      - Laroche, Romain, and Merwan Barlier. "Transfer Reinforcement Learning with Shared Dynamics." AAAI. 2017.
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, source_datasets, verbose = False, **regressor_params):
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, verbose)
        
        self.n_source_mdps = len(source_datasets)
        source_data = np.concatenate(source_datasets)
        self._source_sa = source_data[:,1:self._r_idx]
        self._source_s_prime = source_data[:,self._s_idx:-1]
        
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 0 else Uniform(self._actions)
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        self._iteration = 0
        
        self._iter(data[:,1:self._r_idx], data[:,self._r_idx:self._s_idx], data[:,self._s_idx:-1], data[:,-1], **kwargs)
        sa = np.concatenate((data[:,1:self._r_idx], self._source_sa))
        r = self._policy.Q.values(sa)
        s_prime = np.concatenate((data[:,self._s_idx:-1], self._source_s_prime))
        absorbing = np.concatenate((data[:,-1], np.zeros(self._source_sa.shape[0])))
        
        for _ in range(self._max_iterations-1):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            
        self._result.update_step(n_episodes = self.n_episodes, n_samples = data.shape[0], n_eff = sa.shape[0])
    
    def reset(self):
        
        super().reset()
        
        self._result.add_fields(n_source_mdps = self.n_source_mdps, n_source_samples = self._source_sa.shape[0])
    