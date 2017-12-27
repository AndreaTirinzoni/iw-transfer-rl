import numpy as np

class QFunction:
    """
    Base class for all Q-functions
    """

    def __call__(self, state, action):
        """
        Computes the value of the given state-action couple
        
        Parameters
        ----------
        state: an S-dimensional vector
        action: an A-dimensional vector
        
        Returns
        -------
        The value of (state,action).
        """
        
    def values(self, sa):
        """
        Computes the values of all state-action vectors in sa
        
        Parameters
        ----------
        sa: an Nx(S+A) matrix
        
        Returns
        -------
        An N-dimensional vector with the value of each state-action row vector in sa
        """

class FittedQ(QFunction):
    """
    A FittedQ is a Q-function represented by an underlying regressor
    that has been fitted on some data. The regressor receives SA-dimensional
    vectors and predicts their scalar value.
    This should be prefered for continuous action-spaces. For discrete action-spaces
    use DiscreteFittedQ instead.
    """
    
    def __init__(self, regressor, state_dim, action_dim):
        
        assert hasattr(regressor, "predict")
        self._regressor = regressor
        self._state_dim = state_dim
        self._action_dim = action_dim
        
    def __call__(self, state, action):
        
        return self.values(np.concatenate((state,action),0)[np.newaxis,:])
    
    def values(self, sa):
        
        if not np.shape(sa)[1] == self._state_dim + self._action_dim:
            raise AttributeError("An Nx(S+A) matrix must be provided")
        return self._regressor.predict(sa)

class DiscreteFittedQ(QFunction):
    """
    A DiscreteFittedQ is a Q-function represented by a set of underlying regressors,
    one for each discrete action. This is only for discrete action-spaces.
    """
    
    def __init__(self, regressor_list, state_dim):
        
        for r in regressor_list:
            assert hasattr(r, "predict")
            
        self._regressors = regressor_list
        self._state_dim = state_dim
        self._n_actions = len(regressor_list)
        
    def __call__(self, state, action):
        
        if not np.shape(state)[0] == self._state_dim:
            raise AttributeError("State is not of the right shape")
        return self._action_values(state[np.newaxis,:], action)
    
    
    def _action_values(self, states, action):
        
        return self._regressors[action].predict(states)
    
    def values(self, sa):
        
        if not np.shape(sa)[1] == self._state_dim + 1:
            raise AttributeError("An Nx(S+1) matrix must be provided")
        
        vals = np.zeros(np.shape(sa)[0])
        check_mask = np.zeros(np.shape(sa)[0])
        
        for a in range(self._n_actions):
            mask = sa[:,-1] == a
            check_mask = np.logical_or(mask,check_mask)
            vals[mask] = self._action_values(sa[mask, 0:-1], a)
        
        if not np.all(check_mask):
            raise AttributeError("Some action in sa does not exist")
        
        return vals
    
    
    