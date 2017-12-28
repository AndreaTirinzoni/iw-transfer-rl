class Algorithm(object):
    """
    Base class for all algorithms
    """
    
    def __init__(self, mdp, policy, n_episodes, verbose = False):
        
        self._mdp = mdp
        self.policy = policy
        self._n_episodes = n_episodes
        self.verbose = verbose
        
    def display(self, msg):
        """
        Displays the given message if verbose is True.
        """
        
        if self.verbose:
            print(msg)
            
    def step(self):
        """
        Performs a training step. This varies based on the algorithm.
        Tipically, one or more episodes are collected and the internal structures are accordingly updated.
        """
        raise NotImplementedError