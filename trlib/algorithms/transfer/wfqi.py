import numpy as np
import scipy.stats as stats
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from trlib.policies.policy import Uniform
from trlib.utilities.interaction import generate_episodes, split_data
from copy import deepcopy

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
        noise = np.sqrt(gp.kernel_.get_params()['k2__noise_level'])
        if noise <= np.min(std_gp):
            std_gp = std_gp - noise
        else:
            std_gp = std_gp - np.min(std_gp)
            print("WARNING: GP noise is greater than prediction variances")
    return mu_gp, std_gp
    
class WFQI(FQI):
    """
    Importance Weighted Fitted Q-Iteration (IWFQI)

    References
    ----------

      - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
        Importance Weighted Transfer of Samples in Reinforcement Learning.
        International Conference on Machine Learning. 2018.
    """
    
    def __init__(self, mdp, policy, actions, batch_size, max_iterations, regressor_type, source_datasets, var_rw, var_st, max_gp,
                 weight_estimator = estimate_weights_mean, max_weight = 1000, kernel_rw = None, kernel_st = None, weight_rw = True, weight_st = [True],
                 subtract_noise_rw = False, subtract_noise_st = False, wr = None, ws = None, init_policy = None, verbose = False, **regressor_params):
        
        self._var_rw = var_rw
        self._var_st = var_st
        self._max_gp = max_gp
        self._weight_estimator = weight_estimator
        self._max_weight = max_weight
        self._kernel_rw = kernel_rw
        
        if isinstance(kernel_st, list):
            assert len(kernel_st) == mdp.state_dim
            self._kernel_st = kernel_st
        else:
            self._kernel_st = [deepcopy(kernel_st) for _ in range(mdp.state_dim)]
        
        self._weight_rw = weight_rw
        self._weight_st = weight_st
        self._subtract_noise_rw = subtract_noise_rw
        self._subtract_noise_st = subtract_noise_st
        self._n_source_mdps = len(source_datasets)
        self._wr = wr
        self._ws = ws
        
        super().__init__(mdp, policy, actions, batch_size, max_iterations, regressor_type, init_policy, verbose, **regressor_params)
        
        self._source_predictions_rw = []
        self._source_predictions_st = []
        self._source_sa = []
        self._source_r = []
        self._source_s_prime = []
        self._source_absorbing = []
        
        for data in source_datasets:
            
            _,_,_,r,s_prime,absorbing,sa = split_data(data[0], self._mdp.state_dim, self._mdp.action_dim)
            self._source_sa.append(sa)
            self._source_r.append(r)
            self._source_s_prime.append(s_prime)
            self._source_absorbing.append(absorbing)
            self._source_predictions_rw.append(data[1])
            self._source_predictions_st.append(data[2])
    
    def _get_weighted_rw(self, target_sa, target_r, target_absorbing):
        
        w_r = []
        w_r.append(np.ones(target_sa.shape[0]))
        
        if self._wr is not None:
            w_r.append(self._wr)
        else:
            if self._weight_rw:
                self.display("Fitting reward GP")
                gp_r = _fit_gp(target_sa, target_r, self._kernel_rw, self._max_gp)
            
            for k in range(self._n_source_mdps):
                
                if self._weight_rw:
                    self.display("Predicting reward GP for source " + str(k))
                    mu_gp_t, std_gp_t = _predict_gp(gp_r, self._source_sa[k], subtract_noise = self._subtract_noise_rw)
                    mu_gp_s, std_gp_s = self._source_predictions_rw[k]
                    w_r.append(self._weight_estimator(self._source_r[k], mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, self._var_rw, self._max_weight))
                else:
                    w_r.append(np.ones(self._source_r[k].shape[0]))
        
        w_r = np.concatenate(w_r, axis = 0)
        
        source_sa = np.concatenate(self._source_sa, axis = 0)
        sa = np.concatenate((target_sa,source_sa), axis = 0)
        
        source_absorbing = np.concatenate(self._source_absorbing, axis = 0)
        absorbing = np.concatenate((target_absorbing,source_absorbing), axis = 0)
        
        source_r = np.concatenate(self._source_r, axis = 0)
        r = np.concatenate((target_r,source_r), axis = 0)
        
        return sa, r, absorbing, w_r
    
    def _get_weighted_st(self, target_sa, target_s_prime, target_absorbing):
        
        if self._ws is not None:
            w_s = []
            w_s.append(np.ones(target_sa.shape[0]))
            w_s.append(self._ws)
            w_s = np.concatenate(w_s, axis = 0)
        else:
            w_s = 1
            
            for d in range(self._mdp.state_dim):
                
                if self._weight_st[d]:
                    self.display("Fitting transition GP " + str(d))
                    y = target_s_prime if target_s_prime.ndim == 1 else target_s_prime[:,d]
                    gp_s = _fit_gp(target_sa, y, self._kernel_st[d], self._max_gp)
                    
                w = []
                w.append(np.ones(target_sa.shape[0]))
                
                for k in range(self._n_source_mdps):
                    
                    if self._weight_st[d]:
                        self.display("Predicting transition GP " + str(d) + " for source " + str(k))
                        mu_gp_t, std_gp_t = _predict_gp(gp_s, self._source_sa[k], subtract_noise = self._subtract_noise_st)
                        mu_gp_s, std_gp_s = self._source_predictions_st[k][d]
                        samples = self._source_s_prime[k] if self._source_s_prime[k].ndim == 1 else self._source_s_prime[k][:,d]
                        w.append(self._weight_estimator(samples, mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, self._var_st, self._max_weight))
                    else:
                        w.append(np.ones(self._source_s_prime[k].shape[0]))
                
                w = np.concatenate(w, axis = 0)
                w_s *= w
        
        source_sa = np.concatenate(self._source_sa, axis = 0)
        sa = np.concatenate((target_sa,source_sa), axis = 0)
        
        source_absorbing = np.concatenate(self._source_absorbing, axis = 0)
        absorbing = np.concatenate((target_absorbing,source_absorbing), axis = 0)
        
        source_s_prime = np.concatenate(self._source_s_prime, axis = 0)
        s_prime = np.concatenate((target_s_prime,source_s_prime), axis = 0)
        
        return sa, s_prime, absorbing, w_s
    
    def _step_core(self, **kwargs):
        
        policy = self._policy if self._step > 1 else self._init_policy
        self._data.append(generate_episodes(self._mdp, policy, self._batch_size))
        self.n_episodes += self._batch_size
        target_data = np.concatenate(self._data)
        
        self._iteration = 0
        
        _,_,_,target_r,target_s_prime,target_absorbing,target_sa = split_data(target_data, self._mdp.state_dim, self._mdp.action_dim)
        sa, r, absorbing, wr = self._get_weighted_rw(target_sa, target_r, target_absorbing)
        fit_params = {'sample_weight': wr}
        self._iter(sa, r, [], absorbing, **fit_params)
        
        sa, s_prime, absorbing, ws = self._get_weighted_st(target_sa, target_s_prime, target_absorbing)
        r = self._policy.Q.values(sa)
        fit_params = {'sample_weight': ws}
        
        for _ in range(self._max_iterations-1):
            self._iter(sa, r, s_prime, absorbing, **fit_params)
            
        wr_mean = np.mean(wr)
        ws_mean = np.mean(ws)
        wr_mean2 = np.mean(np.multiply(wr,wr))
        ws_mean2 = np.mean(np.multiply(ws,ws))
        wr_eff = wr.shape[0] * wr_mean ** 2 / wr_mean2
        ws_eff = ws.shape[0] * ws_mean ** 2 / ws_mean2   
        
        self._result.update_step(n_episodes = self.n_episodes, n_target_samples = target_data.shape[0], 
                                 n_source_samples = sa.shape[0] - target_data.shape[0], wr_mean = wr_mean,
                                 ws_mean = ws_mean, wr_eff = wr_eff, ws_eff = ws_eff)
    
    def reset(self):
        
        super().reset()
        
        self._result.add_fields(n_source_mdps = self._n_source_mdps, var_rw = self._var_rw, var_st = self._var_st, max_w = self._max_weight)
    
    