import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from trlib.utilities.data import load_object, save_object
from trlib.utilities.interaction import generate_episodes

def _fit_gp(X, X_train, X_test, y_train, y_test, kernel):
      
    start = time.time()
    
    print("Prior: " + str(kernel))
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    print("Fitting GP")
    gp.fit(X_train,y_train)
    print("Posterior: " + str(gp.kernel_))
    
    if np.shape(X_test)[0] > 0:
        print("Testing")
        y_pred = gp.predict(X_test)
    
        score = mean_squared_error(y_test, y_pred)
        print("MSE: " + str(score))
        
    print("Predicting")
    y_pred, std_pred = gp.predict(X, return_std=True)
    
    del gp
    
    end = time.time()
    
    print("Total time: " + str(end - start))
    
    return (y_pred,std_pred)


def generate_source(mdp, n_episodes, test_fraction, file_name, policy = None, policy_file_name = None, kernel_rw = None, kernel_st = None, load_data = False, fit_rw = True, fit_st = True):
    """
    Generates source data for wfqi and fits the GPs
    
    Parameters
    ----------
    mdp: the MDP to use
    n_episodes: the number of episodes to collect (if load_data is False)
    test_fraction: fraction of the data used for testing the GPs
    file_name: the file where to load/save
    policy: the policy to use
    policy_file_name: the file where to load the policy (ignored if policy is not None)
    kernel_rw: the kernel for fitting the reward GP
    kernel_st: the kernel for fitting the trasition GP
    load_data: whether data should be loaded or generated
    fit_rw: whether the reward should be fitted
    fit_st: whether the state should be fitted
    """
    if load_data:
        print("Loading data")
        data = load_object(file_name)
        source_samples = data[0]
        rw_pred = data[1]
        st_pred = data[2]
    else:
        print("Collecting episodes")
        source_policy = policy if policy is not None else load_object(policy_file_name)
        source_samples = generate_episodes(mdp, source_policy, n_episodes)
        rw_pred = None
        st_pred = None
            
    a_idx = 1 + mdp.state_dim
    r_idx = a_idx + mdp.action_dim
    s_idx = r_idx + 1
    
    X = source_samples[:,1:r_idx]
    
    if fit_rw:
        print("Fitting reward GP")
        y = source_samples[:,r_idx]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_fraction)
        rw_pred = _fit_gp(X, X_train, X_test, y_train, y_test, kernel_rw)
    
    if fit_st:
        st_pred = []
        for d in range(mdp.state_dim):
            print("Fitting transition GP " + str(d))
            y = source_samples[:,(s_idx + d):(s_idx + d + 1)]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_fraction)
            st_pred.append(_fit_gp(X, X_train, X_test, y_train, y_test, kernel_st))
    
    data = [source_samples, rw_pred, st_pred]
    save_object(data, file_name)
