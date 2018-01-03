from trlib.experiments.results import AlgorithmResult
from trlib.policies.policy import Policy
import gym

class Algorithm(object):
    """
    Base class for all algorithms
    """
    
    def __init__(self, name, mdp, policy, verbose = False):
        
        assert isinstance(mdp, gym.Env)
        assert isinstance(policy, Policy)
        
        self._name = name
        self._mdp = mdp
        self._policy = policy
        self._verbose = verbose
        
    def display(self, msg):
        """
        Displays the given message if verbose is True.
        """
        
        if self._verbose:
            print(msg)
            
    def step(self, callbacks = [], **kwargs):
        """
        Performs a training step. This varies based on the algorithm.
        Tipically, one or more episodes are collected and the internal structures are accordingly updated.
        
        Parameters
        ----------
        callbacks: a list of functions to be called with the algorithm as an input after this step
        kwargs: any other algorithm-dependent parameter
        
        Returns
        -------
        A Result object
        """
        self.display("Step {0}".format(self._step))
        
        self._result.add_step(step=self._step)
        
        self._step_core(**kwargs)
        
        for cb in callbacks:
            cb(self)
        
        self._step += 1
        
        return self._result
        
    def _step_core(self, **kwargs):
        """
        The actual algorithm-dependent step. This is automatically called by step() and
        should never be called directly.
        """
        raise NotImplementedError
    
    def run(self, n_steps, callbacks = [], pre_callbacks = [], **kwargs):
        """
        Runs the algorithm for n_steps.
                
        Parameters
        ----------
        n_steps: the number of steps to run
        callbacks: a list of functions to be called with the algorithm as an input after each step
        pre_callbacks: a list of functions to be called before running the algorithm
        kwargs: any other algorithm-dependent parameter
        
        Returns
        -------
        A Result object
        """
        
        for cb in pre_callbacks:
            cb(self)
        
        for _ in range(n_steps):
            self.step(callbacks, **kwargs)

        return self._result
    
    def reset(self):
        """
        Resets the algorithm. Must be called by each overriding method before doing any other operation.
        """
        self._step = 1
        self.n_episodes = 0
        self._result = AlgorithmResult(self._name)