from unittest import TestCase
import numpy as np
from trlib.utilities.interaction import split_data
from trlib.utilities.wfqi_utils import estimate_ideal_weights

class MockMDP():
    
    def __init__(self, coeff):
        self.state_dim = 2
        self.action_dim = 1
        self.coeff = coeff
        
    def get_reward_mean(self, s, a):
        return s[0] + s[1] + a * self.coeff
    
    def get_transition_mean(self, s, a):
        return s + a * self.coeff

class TestUtils(TestCase):
    
    def test_split_data(self):
        
        state_dim = 3
        action_dim = 2
        data = np.array([[0,1,2,3,4,5,6,7,8,9,10],
                         [11,12,13,14,15,16,17,18,19,20,21]])
        
        t,s,a,r,s_prime,absorbing,sa = split_data(data, state_dim, action_dim)
        
        self.assertTrue(np.array_equal(t, np.array([0,11])))
        self.assertTrue(np.array_equal(s, np.array([[1,2,3],[12,13,14]])))
        self.assertTrue(np.array_equal(a, np.array([[4,5],[15,16]])))
        self.assertTrue(np.array_equal(r, np.array([6,17])))
        self.assertTrue(np.array_equal(s_prime, np.array([[7,8,9],[18,19,20]])))
        self.assertTrue(np.array_equal(absorbing, np.array([10,21])))
        self.assertTrue(np.array_equal(sa, np.array([[1,2,3,4,5],[12,13,14,15,16]])))
        
    def test_ideal_weights(self):
        
        target_mdp = MockMDP(1)
        source_mdps = [MockMDP(2), MockMDP(0.5)]
        
        source_data_1 = np.array([[0,1,1,1,4,3,3,0],[0,2,2,-1,2,0,0,0]])
        source_data_2 = np.array([[0,1,1,1,2.5,1.5,1.5,0],[0,2,2,-1,3.5,1.5,1.5,0]])
        source_data = [source_data_1,source_data_2]
        
        var_rw = 0.5 ** 2
        var_st = 0.8 ** 2
        
        wr, ws = estimate_ideal_weights(target_mdp, source_mdps, source_data, var_rw, var_st)
        self.assertTrue(np.all(np.isclose(wr,np.array([0.1353352832366127, 0.1353352832366127, 0.60653065971263342, 0.60653065971263342]))))
        self.assertTrue(np.all(np.isclose(ws[0],np.array([0.45783336177161427*0.45783336177161427]))))
        
        source_mdps = [MockMDP(2), MockMDP(1)]
        wr, ws = estimate_ideal_weights(target_mdp, source_mdps, source_data, var_rw, var_st)
        self.assertTrue(np.all(np.isclose(wr,np.array([0.1353352832366127, 0.1353352832366127, 1.0, 1.0]))))
        self.assertTrue(np.all(np.isclose(ws[0],np.array([0.45783336177161427*0.45783336177161427]))))
        