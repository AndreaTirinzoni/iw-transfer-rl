from unittest import TestCase
import numpy as np
from trlib.policies.policy import Policy
from trlib.utilities.evaluation import _single_eval, evaluate_policy

class MockMDP:
    
    def __init__(self):
        self.gamma = 0.99
        self.horizon = 3
        self.reset()
        
    def reset(self,s = None):
        self._s = np.array(s) if s is not None else np.array([0,0])
        return self._s

    def step(self,a):
        
        self._s[0] += a
        self._s[1] -= a
        
        return self._s, self._s[0], False, {}
    
class MockPolicy(Policy):
    
    def __init__(self,a):
        self._a = a
        
    def sample_action(self, s):
        return self._a
        
        
class TestEvaluation(TestCase):
    
    def test_single_eval(self):
        
        mdp = MockMDP()
        pi = MockPolicy(1)
        
        self.assertEqual(1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, _single_eval(mdp, pi, "discounted", None))
        self.assertEqual(2.0, _single_eval(mdp, pi, "average", None))
        self.assertEqual(11.0 * 1.0 + 12.0 * 0.99 + 13.0 * 0.99 *0.99, _single_eval(mdp, pi, "discounted", np.array([10.0,10.0])))
        self.assertEqual((11.0+12.0+13.0)/3, _single_eval(mdp, pi, "average", np.array([10.0,10.0])))
        
    def test_eval(self):
        
        mdp = MockMDP()
        pi = MockPolicy(1)
        
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 1, initial_states = None, n_threads = 1)
        self.assertEqual(1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, score_mean)
        self.assertEqual(0, score_std)
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 10, initial_states = None, n_threads = 1)
        self.assertEqual(1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, score_mean)
        self.assertEqual(0, score_std)
        
        score_mean, score_std = evaluate_policy(mdp, pi, "average", n_episodes = 1, initial_states = None, n_threads = 1)
        self.assertEqual(2.0, score_mean)
        self.assertEqual(0, score_std)
        score_mean, score_std = evaluate_policy(mdp, pi, "average", n_episodes = 10, initial_states = None, n_threads = 1)
        self.assertEqual(2.0, score_mean)
        self.assertEqual(0, score_std)
        
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 10, initial_states = None, n_threads = 2)
        self.assertEqual(1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, score_mean)
        self.assertEqual(0, score_std)
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 10, initial_states = np.array([10.0,10.0]), n_threads = 1)
        self.assertTrue(np.linalg.norm(11.0 * 1.0 + 12.0 * 0.99 + 13.0 * 0.99 * 0.99 - score_mean) < 0.0000001)
        self.assertTrue(np.linalg.norm(0 - score_std) < 0.0000001)
        
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 10, initial_states = [np.array([0.0,0.0]), np.array([10.0,10.0])], n_threads = 1)
        scores = np.array([1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, 11.0 * 1.0 + 12.0 * 0.99 + 13.0 * 0.99 *0.99])
        self.assertTrue(np.linalg.norm(np.mean(scores) - score_mean) < 0.0000001)
        self.assertTrue(np.linalg.norm(np.std(scores) / np.sqrt(2) - score_std) < 0.0000001)
        
        score_mean, score_std = evaluate_policy(mdp, pi, "discounted", n_episodes = 10, initial_states = [np.array([0.0,0.0]), np.array([10.0,10.0])], n_threads = 2)
        scores = np.array([1.0 * 1.0 + 2.0 * 0.99 + 3.0 * 0.99 *0.99, 11.0 * 1.0 + 12.0 * 0.99 + 13.0 * 0.99 *0.99])
        self.assertTrue(np.linalg.norm(np.mean(scores) - score_mean) < 0.0000001)
        self.assertTrue(np.linalg.norm(np.std(scores) / np.sqrt(2) - score_std) < 0.0000001)
        