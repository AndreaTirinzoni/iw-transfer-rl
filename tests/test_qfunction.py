import numpy as np
from unittest import TestCase
from trlib.policies.qfunction import FittedQ, DiscreteFittedQ

class MockRegressor:
    
    def __init__(self, fun):
        self.fun = fun
    
    def predict(self, X):
        
        return self.fun(X,1)

class TestFittedQ(TestCase):
    
    def test_call(self):
        
        Q = FittedQ(MockRegressor, 3, 2, fun = np.sum)
        
        self.assertEqual(15, Q(np.array([1,2,3]),np.array([4,5])))
        
        with self.assertRaises(AttributeError):
            Q(np.array([1,2,3]), np.array([1]))
            
        with self.assertRaises(AttributeError):
            Q(np.array([1,2,3,4]), np.array([1,2]))
            
    def test_values(self):
        
        Q = FittedQ(MockRegressor, 3, 2, fun = np.sum)
        
        sa = np.array([[1,2,3,4,5], [2,2,3,4,5], [3,2,3,4,5]])
        
        self.assertTrue(np.array_equal(np.array([15,16,17]), Q.values(sa)))
            
        with self.assertRaises(AttributeError):
            Q.values(np.array([[1,2,3,4],[1,2,3,4]]))

    def test_max(self):
        
        Q = FittedQ(MockRegressor, 3, 2, fun = np.sum)
        
        states = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
        actions = [np.array([1,1]), np.array([2,2])]
        absorbing = np.array([0,1,0,1])
        
        vals, acts = Q.max(states, actions)
        self.assertTrue(np.array_equal(np.array([7.0,10.0,13.0,16.0]), vals))
        self.assertTrue(np.array_equal(np.array([[2,2],[2,2],[2,2],[2,2]]), acts))
        
        vals, acts = Q.max(states, actions,absorbing)
        self.assertTrue(np.array_equal(np.array([7.0,0.0,13.0,0.0]), vals))
        self.assertTrue(np.array_equal(np.array([[2,2],[1,1],[2,2],[1,1]]), acts))
        
class TestDiscreteFittedQ(TestCase):
    
    def test_call(self):
        
        Q = DiscreteFittedQ(MockRegressor, 3, [-1,1], fun = np.sum)
        Q._regressors[1] = MockRegressor(np.prod)
        
        self.assertEqual(7, Q(np.array([1,2,4]),-1))
        self.assertEqual(8, Q(np.array([1,2,4]),1))
        
        with self.assertRaises(AttributeError):
            Q(np.array([1,2]), -1)
            
        with self.assertRaises(AttributeError):
            Q(np.array([1,2,3]), 2)
            
    def test_values(self):
        
        Q = DiscreteFittedQ(MockRegressor, 3, [-5,5], fun = np.sum)
        Q._regressors[5] = MockRegressor(np.prod)
        
        sa = np.array([[1,2,3,-5], [2,2,3,5], [3,2,3,-5]])
        
        self.assertTrue(np.array_equal(np.array([6,12,8]), Q.values(sa)))
            
        with self.assertRaises(AttributeError):
            Q.values(np.array([[1,2,3,-5],[1,2,3,2]]))
            
    def test_max(self):
        
        Q = DiscreteFittedQ(MockRegressor, 3, [10,11], fun = np.sum)
        Q._regressors[11] = MockRegressor(np.prod)
        
        states = np.array([[1,1,1],[-2,2,2],[3,3,3],[-4,4,4]])
        absorbing = np.array([1,0,0,1])
        
        vals, acts = Q.max(states)
        self.assertTrue(np.array_equal(np.array([3.0,2.0,27.0,4.0]), vals))
        self.assertTrue(np.array_equal(np.array([10,10,11,10]), acts))
        
        vals, acts = Q.max(states, absorbing=absorbing)
        self.assertTrue(np.array_equal(np.array([0.0,2.0,27.0,0.0]), vals))
        self.assertTrue(np.array_equal(np.array([10,10,11,10]), acts))
        