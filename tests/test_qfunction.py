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
        
        Q = FittedQ(MockRegressor(np.sum), 3, 2)
        
        self.assertEqual(15, Q(np.array([1,2,3]),np.array([4,5])))
        
        with self.assertRaises(AttributeError):
            Q(np.array([1,2,3]), np.array([1]))
            
        with self.assertRaises(AttributeError):
            Q(np.array([1,2,3,4]), np.array([1,2]))
            
    def test_values(self):
        
        Q = FittedQ(MockRegressor(np.sum), 3, 2)
        
        sa = np.array([[1,2,3,4,5], [2,2,3,4,5], [3,2,3,4,5]])
        
        self.assertTrue(np.array_equal(np.array([15,16,17]), Q.values(sa)))
            
        with self.assertRaises(AttributeError):
            Q.values(np.array([[1,2,3,4],[1,2,3,4]]))
        
class TestDiscreteFittedQ(TestCase):
    
    def test_call(self):
        
        rs = [MockRegressor(np.sum), MockRegressor(np.prod)]
        Q = DiscreteFittedQ(rs, 3)
        
        self.assertEqual(7, Q(np.array([1,2,4]),0))
        self.assertEqual(8, Q(np.array([1,2,4]),1))
        
        with self.assertRaises(AttributeError):
            Q(np.array([1,2]), 0)
            
        with self.assertRaises(IndexError):
            Q(np.array([1,2,3]), 2)
            
    def test_values(self):
        
        rs = [MockRegressor(np.sum), MockRegressor(np.prod)]
        Q = DiscreteFittedQ(rs, 3)
        
        sa = np.array([[1,2,3,0], [2,2,3,1], [3,2,3,0]])
        
        self.assertTrue(np.array_equal(np.array([6,12,8]), Q.values(sa)))
            
        with self.assertRaises(AttributeError):
            Q.values(np.array([[1,2,3,0],[1,2,3,2]]))
        