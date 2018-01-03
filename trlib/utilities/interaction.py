import numpy as np
from joblib import Parallel, delayed

def generate_episodes(mdp, policy, n_episodes = 1, n_threads = 1):
    """
    Generates episodes in a given mdp using a given policy
    
    Parameters
    ----------
    mdp: the environment to use
    policy: the policy to use
    n_episodes: the number of episodes to generate
    n_threads: the number of threads to use
    
    Returns
    -------
    A matrix where each row corresponds to a single sample (t,s,a,r,s',absorbing)
    """
    
    if n_threads == 1:
        episodes = [_single_episode(mdp, policy) for _ in range(n_episodes)]
    elif n_threads > 1:
        episodes = Parallel(n_jobs = n_threads)(delayed(_single_episode)(mdp, policy) for _ in range(n_episodes))
        
    return np.concatenate(episodes)
    

def _single_episode(mdp, policy):
    
    episode = np.zeros((mdp.horizon, 1 + mdp.state_dim + mdp.action_dim + 1 + mdp.state_dim + 1))
    a_idx = 1 + mdp.state_dim
    r_idx = a_idx + mdp.action_dim
    s_idx = r_idx + 1
    
    s = mdp.reset()
    t = 0
    
    while t < mdp.horizon:
    
        episode[t,0] = t
        episode[t,1:a_idx] = s
        
        a = policy.sample_action(s)
        s,r,done,_ = mdp.step(a)
        episode[t,a_idx:r_idx] = a
        episode[t,r_idx] = r
        episode[t,s_idx:-1] = s
        episode[t,-1] = 1 if done else 0
        
        t += 1
        if done:
            break
    
    return episode[0:t,:]

def split_data(data, state_dim, action_dim):
    """
    Splits the data into (t,s,a,r,s_prime,absorbing,sa)
    """
    
    assert data.shape[1] == 3 + 2*state_dim + action_dim
    
    a_idx = 1 + state_dim
    r_idx = a_idx + action_dim
    s_idx = r_idx + 1
    
    t = data[:,0]
    s = data[:,1:a_idx].squeeze()
    a = data[:,a_idx:r_idx].squeeze()
    r = data[:,r_idx]
    s_prime = data[:,s_idx:-1].squeeze()
    absorbing = data[:,-1]
    sa = data[:,1:r_idx]
    
    return t,s,a,r,s_prime,absorbing,sa
    