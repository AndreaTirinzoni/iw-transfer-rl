import numpy as np
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.utilities.interaction import generate_episodes, split_data
from trlib.policies.policy import Uniform

class Laroche2017(FQI):
    """
    Transfer Reinforcement Learning with Shared Dynamics
    
    References
    ----------
    
      - Laroche, Romain, and Merwan Barlier. "Transfer Reinforcement Learning with Shared Dynamics." AAAI. 2017.
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, source_datasets, init_policy = None, verbose = False, **regressor_params):
        
        self._n_source_mdps = len(source_datasets)
        source_data = np.concatenate(source_datasets)
        _,_,_,_,self._source_s_prime,self._source_absorbing,self._source_sa = split_data(source_data, mdp.state_dim, mdp.action_dim)
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, init_policy, verbose, **regressor_params)
        
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        data = np.concatenate(self._data)
        
        self._iteration = 0
        _,_,_,r,s_prime,absorbing,sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)
        self._iter(sa, r, s_prime, absorbing, **kwargs)
        
        sa = np.concatenate((sa, self._source_sa))
        r = self._policy.Q.values(sa)
        s_prime = np.concatenate((s_prime, self._source_s_prime))
        absorbing = np.concatenate((absorbing, self._source_absorbing))
        
        for _ in range(self._max_iterations-1):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            
        self._result.update_step(n_episodes = self.n_episodes, n_target_samples = data.shape[0], n_source_samples = self._source_sa.shape[0], n_eff = sa.shape[0])
    
    def reset(self):
        
        super().reset()
        
        self._result.add_fields(n_source_mdps = self._n_source_mdps, n_source_samples = self._source_sa.shape[0])
    