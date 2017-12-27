import numpy as np
from unittest import TestCase
from trlib.policies.valuebased import ValueBased, Softmax
from trlib.policies.qfunction import QFunction
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.parametric import Parametric, Gaussian

class TestQFunction(QFunction):
    
    def __init__(self,vals):
        self.vals = vals
        
    def __call__(self, state):
        
        return np.array(self.vals)

class TestValueBased(TestCase):
    
    def test_actions(self):
        
        actions = [1,2,3]
        Q = QFunction()
        pi = ValueBased(actions,Q)

        self.assertEqual(actions, pi.actions.tolist())
            
        with self.assertRaises(AttributeError):
            pi.actions = [4,5,6]
            
        with self.assertRaises(AttributeError):
            del pi.actions
            
    def test_Q(self):
        
        actions = [1,2,3]
        Q = QFunction()
        pi = ValueBased(actions,Q)
        
        self.assertEqual(Q, pi.Q)
        
        with self.assertRaises(TypeError):
            pi.Q = [4,5,6]
            
        self.assertEqual(Q, pi.Q)
        pi.Q = QFunction()
        self.assertNotEqual(Q, pi.Q)
            
        with self.assertRaises(AttributeError):
            del pi.Q
            
class TestEpsilonGreedy(TestCase):
        
    def test_epsilon(self):

        actions = [10,100,1000]
        pi = EpsilonGreedy(actions,QFunction(),0.5)
        
        self.assertEqual(0.5,pi.epsilon)
        pi.epsilon = 0.6
        self.assertEqual(0.6, pi.epsilon)
        
        with self.assertRaises(AttributeError):
            pi.epsilon = -1
            
        with self.assertRaises(AttributeError):
            pi.epsilon = 2
            
        with self.assertRaises(AttributeError):
            del pi.epsilon
            
    def test_sample(self):

        actions = [10,100,1000]
        
        pi = EpsilonGreedy(actions,TestQFunction([10,100,1000]),0)
        self.assertEqual(1000, pi.sample_action(0))
        self.assertTrue(np.linalg.norm(np.array([0.,0.,1.]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.3
        self.assertTrue(np.linalg.norm(np.array([0.1,0.1,0.8]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.6
        np.random.seed(0)
        self.assertEqual(100, pi.sample_action(0))
        
        pi = EpsilonGreedy(actions,TestQFunction([10,1000,100]),0)
        self.assertEqual(100, pi.sample_action(0))
        self.assertTrue(np.linalg.norm(np.array([0.,1.,0.]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.3
        self.assertTrue(np.linalg.norm(np.array([0.1,0.8,0.1]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.6
        np.random.seed(0)
        self.assertEqual(100, pi.sample_action(0))
        
        pi = EpsilonGreedy(actions,TestQFunction([1000,100,10]),0)
        self.assertEqual(10, pi.sample_action(0))
        self.assertTrue(np.linalg.norm(np.array([1.,0.,0.]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.3
        self.assertTrue(np.linalg.norm(np.array([0.8,0.1,0.1]) - pi(0)) < 0.0000001)
        pi.epsilon = 0.6
        np.random.seed(0)
        self.assertEqual(100, pi.sample_action(0))
        
class TestSoftmax(TestCase):

    def test_tau(self):
        
        actions = [10,100,1000]
        pi = Softmax(actions,TestQFunction([1,2,3]),1.0)
        
        self.assertEqual(1.0, pi.tau)
        pi.tau = 100.0
        self.assertEqual(100.0, pi.tau)
        pi.tau = 0.5
        self.assertEqual(0.5, pi.tau)
        
        with self.assertRaises(AttributeError):
            pi.tau = 0
        self.assertEqual(0.5, pi.tau)
        
        with self.assertRaises(AttributeError):
            pi.tau = -1
        self.assertEqual(0.5, pi.tau)
        
        with self.assertRaises(AttributeError):
            del pi.tau
        self.assertEqual(0.5, pi.tau)
        
    def test_sample(self):
        
        actions = [10,100,1000]
        
        pi = Softmax(actions,TestQFunction([1,2,3]),1.0)
        probs = np.exp([1,2,3])
        probs /= np.sum(probs)
        self.assertTrue(np.linalg.norm(probs - pi(0)) < 0.0000001)
        np.random.seed(0)
        a = np.random.choice(actions,p = probs)
        np.random.seed(0)
        self.assertEqual(a, pi.sample_action(0))
        
        pi = Softmax(actions,TestQFunction([1,2,3]),100.0)
        probs = np.exp(np.array([1,2,3]) / 100.0)
        probs /= np.sum(probs)
        self.assertTrue(np.linalg.norm(probs - pi(0)) < 0.0000001)
        np.random.seed(0)
        a = np.random.choice(actions,p = probs)
        np.random.seed(0)
        self.assertEqual(a, pi.sample_action(0))
        
        pi = Softmax(actions,TestQFunction([1,2,3]), 0.1)
        probs = np.exp(np.array([1,2,3]) * 10.0)
        probs /= np.sum(probs)
        self.assertTrue(np.linalg.norm(probs - pi(0)) < 0.0000001)
        np.random.seed(0)
        a = np.random.choice(actions,p = probs)
        np.random.seed(0)
        self.assertEqual(a, pi.sample_action(0))

class TestParametric(TestCase):
    
    def test_theta(self):
        
        pi = Parametric(np.array([1,2,3]))
        
        self.assertTrue(np.array_equal(np.array([1,2,3]), pi.theta))
        pi.theta = np.array([4,5,6])
        self.assertTrue(np.array_equal(np.array([4,5,6]), pi.theta))
        
        with self.assertRaises(AttributeError):
            pi.theta = [7,8,9]
        self.assertTrue(np.array_equal(np.array([4,5,6]), pi.theta))
        
        with self.assertRaises(AttributeError):
            pi.theta = np.array([7,8,9,10])
        self.assertTrue(np.array_equal(np.array([4,5,6]), pi.theta))
        
        with self.assertRaises(AttributeError):
            del pi.theta
        self.assertTrue(np.array_equal(np.array([4,5,6]), pi.theta))

class TestGaussian(TestCase):
    
    def test_sigma(self):
        
        pi = Gaussian(np.array([1,2,3]), 1.0, None)
        
        self.assertEqual(1.0, pi.sigma)
        pi.sigma = 0.1
        self.assertEqual(0.1, pi.sigma)
                         
        with self.assertRaises(AttributeError):
            pi.sigma = 0
        self.assertEqual(0.1, pi.sigma)
        
        with self.assertRaises(AttributeError):
            pi.sigma = -10
        self.assertEqual(0.1, pi.sigma)
        
        with self.assertRaises(AttributeError):
            del pi.sigma
        self.assertEqual(0.1, pi.sigma)
                         
    def test_sample(self):
        
        phi = lambda s: s
        sigma = 0.1
        theta = np.array([1,1])
        states = [np.array([1,1]), np.array([0,0]), np.array([2,-3])]
        means = [2,0,-1]
        
        pi = Gaussian(theta, sigma, phi)
        
        for state,mn in zip(states,means):
            for a in range(-2,3):
                pdf = 1.0 / np.sqrt(2*np.pi*sigma**2) * np.exp(- (a - np.dot(theta,phi(state)))**2 / (2*sigma**2))
                self.assertTrue(np.linalg.norm(pdf - pi(state,a)) < 0.0000001)
                grad = (a - np.dot(theta,phi(state))) * phi(state) / sigma ** 2
                self.assertTrue(np.linalg.norm(grad - pi.log_gradient(state,a)) < 0.0000001)
            
            np.random.seed(0)
            x = np.zeros(1000)
            for i in range(1000):
                x[i] = pi.sample_action(state)
            self.assertTrue(np.linalg.norm(mn-np.mean(x)) < 0.01)

                         