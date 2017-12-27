import numpy as np

class Policy(object):
    """
    Base class for all policies.
    """
    
    def sample_actions(self, states):
        """
        Samples actions in the given states.
        
        Parameters
        ----------
        states: an NxS matrix, where N is the number of states and S is the state-space dimension
          
        Returns
        -------
        An NxA matrix, where N is the number of states and A is the action-space dimension
        """
        raise NotImplementedError
    
    def sample_action(self, state):
        """
        Samples an action in the given state.
        
        Parameters
        ----------
        state: an S-dimensional vector, where S is the state-space dimension
          
        Returns
        -------
        An A-dimensional vector, where A is the action-space dimension
        """
        raise NotImplementedError