import numpy as np
from numpy import matlib
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes, split_data

def _distance(x, y):
    if x.ndim == 1:
        return np.linalg.norm(x[:,np.newaxis]-y[:,np.newaxis], axis = 1)
    return np.linalg.norm(x-y, axis = 1)

def _phi(x, delta):
    return np.exp(- np.square(x) / delta)

def _weights(target_sa, source_sa, delta_sa):
        
    n_target = target_sa.shape[0]
    n_source = source_sa.shape[0]

    sa_t = matlib.repeat(target_sa, n_source, axis = 0)
    sa_s = matlib.repmat(source_sa, n_target, 1)
    
    dist = _distance(sa_t, sa_s)
    w = _phi(dist, delta_sa)
    w = w.reshape(n_target,-1)
    w /= np.sum(w,1)[:, np.newaxis]
    w[np.isnan(w)] = 0
    dist = dist.reshape(n_target,-1)
    
    return w, dist

def _compliance(w, target_s, target_r, target_s_prime, source_s, source_r, source_s_prime, delta_s_prime, delta_r):
    
    n_target = target_r.shape[0]
    n_source = source_r.shape[0]
    
    s_prime_t = matlib.repeat(target_s_prime, n_source, axis = 0).squeeze()
    s_prime_s = matlib.repmat(source_s_prime, n_target, 1).reshape(n_target*n_source,-1).squeeze()
    s_t = matlib.repeat(target_s, n_source, axis = 0).squeeze()
    s_s = matlib.repmat(source_s, n_target, 1).reshape(n_target*n_source,-1).squeeze()
    
    phi = _phi(_distance(s_prime_t, s_t + (s_prime_s - s_s)), delta_s_prime)
    lambda_p = np.multiply(w, phi.reshape(n_target,-1))
    
    r_t = matlib.repeat(target_r, n_source, axis = 0).squeeze()
    r_s = matlib.repmat(source_r, n_target, 1).reshape(n_target*n_source,-1).squeeze()
    
    phi = _phi(_distance(r_t, r_s), delta_r)
    lambda_r = np.multiply(w, phi.reshape(n_target,-1))
    
    return np.mean(lambda_p, 1) * np.mean(lambda_r, 1)

def _avg_distances(w, distances, mu):
    
    idx = np.argsort(w, axis = 1)
    w = w[np.arange(w.shape[0])[:,np.newaxis],idx]
    distances = distances[np.arange(w.shape[0])[:,np.newaxis],idx]
    
    w_cum = np.cumsum(w,1)
    mask = w_cum > mu
    w_cum[mask] = 0
    idx = np.argmax(w_cum, axis = 1)
    distances[mask] = 0
    
    return np.sum(distances, 1) / (idx + 1)

def _relevance(lambdas, avg_distances):
    
    lambdas = lambdas / np.sum(lambdas)
    return np.exp(- np.square((lambdas - 1) / avg_distances))

def _compliance_relevance(target_sa, target_s, target_r, target_s_prime, source_sa, source_s, source_r, 
                          source_s_prime, prior, delta_sa, delta_s_prime, delta_r, mu):
    
    w, _ = _weights(target_sa, source_sa, delta_sa)
    lambda_t = _compliance(w, target_s, target_r, target_s_prime, source_s, source_r, source_s_prime, delta_s_prime, delta_r)
    lambda_t = np.sum(lambda_t) * prior / target_sa.shape[0]
    
    w, dist = _weights(source_sa, target_sa, delta_sa)
    lambda_s = _compliance(w, source_s, source_r, source_s_prime, target_s, target_r, target_s_prime, delta_s_prime, delta_r)
    avg_dist = _avg_distances(w, dist, mu)
    relevance = _relevance(lambda_s, avg_dist)
    
    return lambda_t, relevance

class Lazaric2008(FQI):
    """
    Transfer of Samples in Batch Reinforcement Learning
    
    References
    ----------
    
      - Lazaric, Alessandro, Marcello Restelli, and Andrea Bonarini. 
        "Transfer of samples in batch reinforcement learning." 
        Proceedings of the 25th international conference on Machine learning. ACM, 2008.
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, source_datasets,
                 delta_sa, delta_s_prime, delta_r, mu, n_sample_total, prior = None, init_policy = None, verbose = False, **regressor_params):
        

        self._n_source_mdps = len(source_datasets)
        self._prior = [1 / self._n_source_mdps for _ in range(self._n_source_mdps)] if prior is None else prior
        self._source_data = source_datasets
        self._delta_sa = delta_sa
        self._delta_s_prime = delta_s_prime
        self._delta_r = delta_r
        self._mu = mu
        self._n_sample_total = n_sample_total
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, init_policy, verbose, **regressor_params)
    
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        target_data = np.concatenate(self._data)
        
        self.display("Computing compliances and relevancies")
        
        _,target_s,_,target_r,target_s_prime,_,target_sa = split_data(target_data, self._mdp.state_dim, self._mdp.action_dim)
        compliances = []
        relevances = []
        for i in range(self._n_source_mdps):
            
            _,source_s,_,source_r,source_s_prime,_,source_sa = split_data(self._source_data[i], self._mdp.state_dim, self._mdp.action_dim)
            comp,rel = _compliance_relevance(target_sa, target_s, target_r, target_s_prime, source_sa,
                                             source_s, source_r, source_s_prime, self._prior[i], 
                                             self._delta_sa, self._delta_s_prime, self._delta_r, self._mu)
            compliances.append(comp)
            relevances.append(rel / np.sum(rel))
            
        compliances = np.array(compliances)
        compliances /= np.sum(compliances)
        
        self.display("Sampling source data")
        
        data = target_data
        
        for i in range(self._n_source_mdps):
            
            n = self._n_sample_total * compliances[i]
            n = int(n)
            idx = np.random.choice(self._source_data[i].shape[0], size = n, replace = True, p = relevances[i])
            data = np.concatenate((data, self._source_data[i][idx,:]))
        
        self._iteration = 0
        
        _,_,_,r,s_prime,absorbing,sa = split_data(data, self._mdp.state_dim, self._mdp.action_dim)
        
        for _ in range(self._max_iterations):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            
        self._result.update_step(n_episodes = self.n_episodes, n_eff = data.shape[0], n_target_samples = target_data.shape[0],
                                 n_source_samples = data.shape[0] - target_data.shape[0], compliances = compliances.tolist())
    
    def reset(self):
        
        super().reset()
        
        self._result.add_fields(n_source_mdps = self._n_source_mdps, n_sample_total = self._n_sample_total)
    
    
        