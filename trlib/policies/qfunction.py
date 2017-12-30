import numpy as np
from numpy import matlib
from examples.test import regressor_params

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
        raise NotImplementedError
        
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
        raise NotImplementedError
    
    def max(self, states, actions=None, absorbing=None):
        """
        Computes the action among actions achieving the maximum value for each state in states
        
        Parameters:
        -----------
        states: an NxS matrix
        actions: a list of A-dimensional vectors
        absorbing: an N-dimensional vector specifying whether each state is absorbing
        
        Returns:
        --------
        An NxA matrix with the maximizing actions and an N-dimensional vector with their values
        """
        raise NotImplementedError
    
class ZeroQ(QFunction):
    
    def __call__(self, state, action):
        return 0
    
    def values(self, sa):
        return np.zeros(np.shape(sa)[0])
        

class FittedQ(QFunction):
    """
    A FittedQ is a Q-function represented by an underlying regressor
    that has been fitted on some data. The regressor receives SA-dimensional
    vectors and predicts their scalar value.
    This should be prefered for continuous action-spaces. For discrete action-spaces
    use DiscreteFittedQ instead.
    """
    
    def __init__(self, regressor_type, state_dim, action_dim, **regressor_params):
        
        self._regressor = regressor_type(**regressor_params)
        self._state_dim = state_dim
        self._action_dim = action_dim
        
    def __call__(self, state, action):
        
        return self.values(np.concatenate((state,action),0)[np.newaxis,:])
    
    def values(self, sa):
        
        if not np.shape(sa)[1] == self._state_dim + self._action_dim:
            raise AttributeError("An Nx(S+A) matrix must be provided")
        return self._regressor.predict(sa)
    
    def max(self, states, actions=None, absorbing=None):
        
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        if actions is None:
            raise AttributeError("Actions must be provided")

        n_actions = len(actions)
        n_states = np.shape(states)[0]
        actions = np.array(actions).reshape((n_actions,self._action_dim))
        
        sa = np.empty((n_states * n_actions, self._state_dim + self._action_dim))
        for i in range(n_states):
            sa[i*n_actions:(i+1)*n_actions,0:self._state_dim] = matlib.repmat(states[i,:], n_actions, 1)
            sa[i*n_actions:(i+1)*n_actions,self._state_dim:] = actions
            
        vals = self.values(sa)
        
        if absorbing is not None:
            absorbing = matlib.repmat(absorbing,n_actions,1).T.flatten()
            vals[absorbing == 1] = 0
          
        max_vals = np.empty(n_states)
        max_actions = np.empty((n_states,self._action_dim))
        
        for i in range(n_states):
            val = vals[i*n_actions:(i+1)*n_actions]
            a = np.argmax(val)
            max_vals[i] = val[a]
            max_actions[i,:] = actions[a,:]
            
        return (max_vals,max_actions)
    
    def fit(self, sa, q, **fit_params):
        
        self._regressor.fit(sa, q, **fit_params)
        
class DiscreteFittedQ(QFunction):
    """
    A DiscreteFittedQ is a Q-function represented by a set of underlying regressors,
    one for each discrete action. This is only for discrete action-spaces.
    """
    
    def __init__(self, regressor_type, state_dim, n_actions, **regressor_params):
            
        self._regressors = [regressor_type(**regressor_params) for _ in range(n_actions)]
        self._state_dim = state_dim
        self._n_actions = n_actions
        
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
    
    def max(self, states, actions=None, absorbing=None):
        
        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
    
        n_states = np.shape(states)[0]
        vals = np.empty((n_states,self._n_actions))
        
        for a in range(self._n_actions):
            vals[:,a] = self._action_values(states, a)
        
        if absorbing is not None:
            vals[absorbing == 1, :] = 0
            
        max_actions = np.argmax(vals,1)
        idx = np.ogrid[:n_states]
        max_vals = vals[idx,max_actions]
        
        return max_vals, max_actions
    
    def fit(self, sa, q, **fit_params):
        
        for a in range(self._n_actions):
            mask = sa[:,-1] == a
            self._regressors[a].fit(sa[mask, 0:-1], q[mask], **fit_params)
    