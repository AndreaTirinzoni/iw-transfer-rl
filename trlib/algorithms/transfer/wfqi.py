import numpy as np
import scipy.stats as stats
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes

def estimate_weights_mean(samples, mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, noise, max_weight):
    
    n_samples = samples.shape[0]
    var_num = noise + std_gp_t ** 2
    var_denom = noise - std_gp_s ** 2
    w = np.zeros(n_samples)
    
    for i in range(n_samples):
        
        if var_denom[i] > 0:
            num = stats.norm.pdf(samples[i], mu_gp_t[i], np.sqrt(var_num[i]))
            denom = stats.norm.pdf(samples[i], mu_gp_s[i], np.sqrt(var_denom[i]))
            w[i]= (num / denom) * (noise / var_denom[i])
            w[i] = min(w[i], max_weight)
        else:
            print("WARNING: discarding sample due to imprecise GP")
    
    return w

def estimate_weights_heuristic(samples, mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, noise, max_weight):
    
    n_samples = samples.shape[0]
    var_num = noise + std_gp_t ** 2
    var_denom = noise + std_gp_s ** 2
    w = np.zeros(n_samples)
    
    for i in range(n_samples):
        
        num = stats.norm.pdf(samples[i], mu_gp_t[i], np.sqrt(var_num[i]))
        denom = stats.norm.pdf(samples[i], mu_gp_s[i], np.sqrt(var_denom[i]))
        w[i]= num / denom
        w[i] = min(w[i], max_weight)
    
    return w

def _fit_gp(X, y, kernel, max_gp):

    n_samples = X.shape[0]

    n_gp = min(n_samples, max_gp)
    
    X = X[(n_samples-n_gp):n_samples,:]
    y = y[(n_samples-n_gp):n_samples]
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)
    gp.fit(X,y)
    
    return gp

def _predict_gp(gp, X, subtract_noise = False):
    
    mu_gp, std_gp = gp.predict(X, return_std=True)
    if subtract_noise:
        std_gp = std_gp - np.min(std_gp)
    return mu_gp, std_gp

class WFQI(FQI):
    """
    Weighted Fitted Q-Iteration
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, source_datasets, var_rw, var_st, max_gp,
                 max_weight = 1000, kernel_rw = None, kernel_st = None, verbose = False, **regressor_params):
        
        self._var_rw = var_rw
        self._var_st = var_st
        self._max_gp = max_gp
        self._max_weight = max_weight
        self._kernel_rw = kernel_rw
        self._kernel_st = kernel_st
        self.n_source_mdps = len(source_datasets)
        
        self._source_predictions_rw = []
        self._source_predictions_st = []
        self._source_samples = []
        
        for data in source_datasets:
            
            self._source_samples.append(data[0])
            self._source_predictions_rw.append(data[1])
            self._source_predictions_st.append(data[2])
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, verbose, **regressor_params)
    
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
        absorbing = np.concatenate((data[:,-1], self._source_absorbing))
        
        for _ in range(self._max_iterations-1):
            self._iter(sa, r, s_prime, absorbing, **kwargs)
            
        self._result.update_step(n_episodes = self.n_episodes, n_target_samples = data.shape[0], n_source_samples = self._source_sa.shape[0], n_eff = sa.shape[0])
    
    def reset(self):
        
        super().reset()
        
        self._result.add_fields(n_source_mdps = self._n_source_mdps, var_rw = self._var_rw, var_st = self._var_st, max_w = self._max_weight)
    
    