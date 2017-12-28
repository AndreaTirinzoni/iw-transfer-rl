from pytz.tzinfo import _notime
class Algorithm(object):
    """
    Base class for all algorithms
    """
    
    def __init__(self, mdp, policy, max_steps, verbose = False):
        
        self._mdp = mdp
        self.policy = policy
        self._max_steps = max_steps
        self.verbose = verbose
        
    def display(self, msg):
        """
        Displays the given message if verbose is True.
        """
        
        if self.verbose:
            print(msg)
            
    def step(self, callbacks = []):
        """
        Performs a training step. This varies based on the algorithm.
        Tipically, one or more episodes are collected and the internal structures are accordingly updated.
        
        Parameters
        ----------
        callbacks: a list of functions to be called with the algorithm as an input after this step
        """
        raise NotImplementedError
    
    def run(self, callbacks = []):
        """
        Runs the algorithm until the maximum number of steps is reached
                
        Parameters
        ----------
        callbacks: a list of functions to be called with the algorithm as an input after each step
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Resets the algorithm
        """
        raise NotImplementedError