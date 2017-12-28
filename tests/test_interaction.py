from unittest import TestCase
import numpy as np
from trlib.policies.policy import Policy
from trlib.utilities.interaction import generate_episodes, _single_episode

class MockMDP:
    
    def __init__(self):
        self.gamma = 0.99
        self.horizon = 3
        self.state_dim = 3
        self.action_dim = 2
        self.reset()
        
    def reset(self,s = None):
        self._s = np.array(s) if s is not None else np.array([0,0,0])
        return self._s

    def step(self,a):
        
        self._s[0] += a[0]
        self._s[1] += a[1]
        self._s[2] += a[0] + a[1]
        
        r = self._s[0] - self._s[1]
        
        done = self._s[0] > 5
        
        return self._s, r, done, {}
    
class MockPolicy(Policy):
    
    def __init__(self,a):
        self._a = a
        
    def sample_action(self, s):
        return np.array([self._a, s[0]])
    
class TestInteraction(TestCase):
    
    def test_single_episode(self):
        
        mdp = MockMDP()
        
        pi = MockPolicy(1)
        episode = _single_episode(mdp, pi)
        exp_episode = np.array([[0,0,0,0,1,0,1,1,0,1,0],[1,1,0,1,1,1,1,2,1,3,0],[2,2,1,3,1,2,0,3,3,6,0]])
        self.assertTrue(np.array_equal(episode, exp_episode))
        
        pi = MockPolicy(3)
        episode = _single_episode(mdp, pi)
        exp_episode = np.array([[0,0,0,0,3,0,3,3,0,3,0],[1,3,0,3,3,3,3,6,3,9,1]])
        self.assertTrue(np.array_equal(episode, exp_episode))
        
    def test_episodes(self):
        
        mdp = MockMDP()
        
        pi = MockPolicy(1)
        episode = generate_episodes(mdp, pi, 2, 1)
        exp_episode = np.array([[0,0,0,0,1,0,1,1,0,1,0],[1,1,0,1,1,1,1,2,1,3,0],[2,2,1,3,1,2,0,3,3,6,0]])
        exp_episode = np.concatenate((exp_episode,exp_episode))
        self.assertTrue(np.array_equal(episode, exp_episode))
        
        episode = generate_episodes(mdp, pi, 4, 2)
        exp_episode = np.array([[0,0,0,0,1,0,1,1,0,1,0],[1,1,0,1,1,1,1,2,1,3,0],[2,2,1,3,1,2,0,3,3,6,0]])
        exp_episode = np.concatenate((exp_episode,exp_episode,exp_episode,exp_episode))
        self.assertTrue(np.array_equal(episode, exp_episode))
    