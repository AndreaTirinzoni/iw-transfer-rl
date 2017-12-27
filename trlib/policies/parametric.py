import numpy as np
from trlib.policies.policy import Policy
from scipy.stats import norm

class Parametric(Policy):
    """
    A parametric policy is defined by some parameter-vector theta.
    A parametric policy must also define functions for computing its gradient w.r.t. theta.
    """
    
    def __init__(self, theta):
        
        self.theta = theta
        self._theta_dim = np.shape(theta)[0]
        
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self,value):
        if not isinstance(value, np.ndarray):
            raise AttributeError("Theta must be a numpy array")
        if hasattr(self, "_theta") and not np.shape(value) == np.shape(self._theta):
            raise AttributeError("Theta must not change shape")
        self._theta = value
        
    def __call__(self, state, action, theta=None):
        """
        Computes the policy value in the given state, action, and parameter
        
        Parameters
        ----------
        state: S-dimensional vector
        action: A-dimensional vector
        theta: D-dimensional vector (if None, current parameter is used)
        
        Returns
        -------
        The probability pi(a|s;theta)
        """
        raise NotImplementedError
        
    def log_gradient(self, state, action):
        """
        Computes the gradient of the log-policy.
        
        Parameters
        ----------
        state: S-dimensional vector
        action: A-dimensional vector
        
        Returns
        -------
        A D-dimensional vector.
        """    
        raise NotImplementedError

class Gaussian(Parametric):
    """
    The univariate Gaussian policy. 
    This is defined by the parameter vector theta, the standard deviation sigma, and the feature function phi.
    """
    
    def __init__(self, theta, sigma, phi):
        
        super().__init__(theta)
        self.sigma = sigma
        self._phi = phi
        
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self,value):
        if value <= 0:
            raise AttributeError("Sigma must be strictly greater than zero")
        self._sigma = value
        
    def __call__(self, state, action, theta=None):
        
        theta = self._theta if theta is None else theta
        return norm.pdf(action, np.dot(theta,self._phi(state)), self._sigma)
    
    def sample_action(self, state):
        
        return np.array([np.random.normal(np.dot(self._theta,self._phi(state)), self._sigma)])
    
    def log_gradient(self, state, action):
        
        feats = self._phi(state)
        return (action - np.dot(self._theta,feats)) * feats / self._sigma ** 2
        