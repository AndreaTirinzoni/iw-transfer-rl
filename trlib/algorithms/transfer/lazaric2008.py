import numpy as np
from numpy import matlib
from trlib.algorithms.reinforcement.fqi import FQI

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
    dist = dist.reshape(n_target,-1)
    
    return w, dist

def _compliance(w, target_s, target_r, target_s_prime, source_s, source_r, source_s_prime, delta_s_prime, delta_r):
    
    n_target = target_r.shape[0]
    n_source = source_r.shape[0]
    
    s_prime_t = matlib.repeat(target_s_prime, n_source, axis = 0)
    s_prime_s = matlib.repmat(source_s_prime, n_target, 1).reshape(n_target*n_source,)
    s_t = matlib.repeat(target_s, n_source, axis = 0)
    s_s = matlib.repmat(source_s, n_target, 1).reshape(n_target*n_source,)
    
    phi = _phi(_distance(s_prime_t, s_t + (s_prime_s - s_s)), delta_s_prime)
    lambda_p = np.multiply(w, phi.reshape(n_target,-1))
    
    r_t = matlib.repeat(target_r, n_source, axis = 0)
    r_s = matlib.repmat(source_r, n_target, 1).reshape(n_target*n_source,)
    
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
                 delta_sa, delta_s_prime, delta_r, mu, prior = None, verbose = False, **regressor_params):
        

        self._n_source_mdps = len(source_datasets)
        self._prior = [1 / self._n_source_mdps for _ in range(self._n_source_mdps)] if prior is None else prior
        self._source_data = source_datasets
        self._delta_sa = delta_sa
        self._delta_s_prime = delta_s_prime
        self._delta_r = delta_r
        self._mu = mu
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, verbose, **regressor_params)
        
    def _split_data(self, data):
        """
        Splits the data into (sa,r,s_prime,absorbing, s)
        """
        a_idx = 1 + self._mdp.state_dim
        r_idx = a_idx + self._mdp.action_dim
        s_idx = r_idx + 1
        
        return data[:,1:r_idx], data[:,r_idx:s_idx], data[:,s_idx:-1], data[:,-1], data[:,1:a_idx]
        