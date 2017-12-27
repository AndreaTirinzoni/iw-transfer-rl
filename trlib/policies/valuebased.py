import numpy as np
from numpy import matlib
from trlib.policies.policy import Policy
from trlib.policies.qfunction import QFunction

class ValueBased(Policy):
    """
    A value-based policy is a policy that chooses actions based on their value.
    The action-space is always discrete for this kind of policy.
    """
    
    def __init__(self,actions,Q):
        
        self._actions = np.array(actions)
        self._n_actions = len(actions)
        self.Q = Q
        
    @property
    def actions(self):
        return self._actions
    
    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self,value):
        if not isinstance(value, QFunction):
            raise TypeError("The argument must be a QFunction")
        self._Q = value
        
    def __call__(self, state):
        """
        Computes the policy value in the given state
        
        Parameters
        ----------
        state: S-dimensional vector
        
        Returns
        -------
        An A-dimensional vector containing the probabilities pi(.|s)
        """
        raise NotImplementedError
    
    def _q_values(self, state):
        
        return self._Q(np.concatenate((matlib.repmat(state, self._n_actions, 1), self._actions[:,np.newaxis]), 1))
    
    
class EpsilonGreedy(ValueBased):
    """
    The epsilon-greedy policy.
    The parameter epsilon defines the probability of taking a random action.
    Set epsilon to zero to have a greedy policy.
    """
    
    def __init__(self,actions,Q,epsilon):
        
        super().__init__(actions, Q)
        self.epsilon = epsilon
        
    @property
    def epsilon(self):
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self,value):
        if value < 0 or value > 1:
            raise AttributeError("Epsilon must be in [0,1]")
        self._epsilon = value
        
    def __call__(self, state):
        
        probs = np.ones(self._n_actions) * self._epsilon / self._n_actions
        probs[np.argmax(self._Q(state))] += 1 - self._epsilon
        return probs
        
    def sample_action(self, state):
        
        if np.random.uniform() < self._epsilon:
            return np.array([self._actions[np.random.choice(self._n_actions)]])
        else:
            return np.array([self._actions[np.argmax(self._Q(state))]])
        
class Softmax(ValueBased):
    """
    The softmax (or Boltzmann) policy.
    The parameter tau controls exploration (for tau close to zero the policy is almost greedy)
    """
    
    def __init__(self,actions,Q,tau):
        
        super().__init__(actions, Q)
        self.tau = tau
        
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self,value):
        if value <= 0:
            raise AttributeError("Tau must be strictly greater than zero")
        self._tau = value
        
    def __call__(self, state):
        
        exps = np.exp(self._Q(state) / self._tau)
        return exps / np.sum(exps)
        
    def sample_action(self, state):
        
        return np.array([self._actions[np.random.choice(self._n_actions, p = self(state))]])