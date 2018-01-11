import numpy as np
from joblib import Parallel, delayed

def evaluate_policy(mdp, policy, criterion = 'discounted', n_episodes = 1, initial_states = None, n_threads = 1):
    """
    Evaluates a policy on a given MDP.
    
    Parameters
    ----------
    mdp: the environment to use in the evaluation
    policy: the policy to evaluate
    criterion: either 'discounted' or 'average'
    n_episodes: the number of episodes to generate in the evaluation
    initial_states: either None (i), a numpy array (ii), or a list of numpy arrays (iii)
      - (i) initial states are drawn from the MDP distribution
      - (ii) the given array is used as initial state for all episodes
      - (iii) n_episodes is ignored and the episodes are defined by their initial states
    n_threads: the number of threads to use in the evaluation
    
    Returns
    -------
    The mean of the scores and its confidence interval.
    """
    
    assert criterion == 'average' or criterion == 'discounted'
    
    if n_threads == 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = [_single_eval(mdp, policy, criterion, initial_states) for _ in range(n_episodes)]
    elif n_threads > 1 and (initial_states is None or type(initial_states) is np.ndarray):
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, initial_states) for _ in range(n_episodes))
    elif n_threads == 1 and type(initial_states) is list:
        scores = [_single_eval(mdp, policy, criterion, init_state) for init_state in initial_states]
    elif n_threads > 1 and type(initial_states) is list:
        scores = Parallel(n_jobs = n_threads)(delayed(_single_eval)(mdp, policy, criterion, init_state) for init_state in initial_states)
    
    n_episodes = len(initial_states) if type(initial_states) is list else n_episodes
    scores = np.array(scores)
    return np.mean(scores[:,0]), np.std(scores[:,0]) / np.sqrt(n_episodes), np.mean(scores[:,1])

def _single_eval(mdp, policy, criterion, initial_state):
    
    score = 0
    gamma = mdp.gamma if criterion == "discounted" else 1
    
    s = mdp.reset(initial_state)
    t = 0
    
    while t < mdp.horizon:
    
        a = policy.sample_action(s)
        s,r,done,_ = mdp.step(a)
        score += r * gamma**t
        t += 1
        if done:
            break
    
    return score if criterion == "discounted" else score / t, t