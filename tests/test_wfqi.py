from unittest import TestCase
import numpy as np
from trlib.algorithms.transfer.wfqi import estimate_weights_mean,\
    estimate_weights_heuristic

class TestWFQI(TestCase):
    
    def test_weight_mean(self):
        
        samples = np.array([0.0, 1.0, 2.0])
        mu_gp_t = np.array([0.5, 0.8, 1.2])
        std_gp_t = np.sqrt(np.array([0.1, 0.2, 0.3]))
        mu_gp_s = np.array([0.0, 1.0, 2.0])
        std_gp_s = np.sqrt(np.array([0.01, 0.02, 0.03]))
        noise = 2.0
        max_weight = 1000
        
        w = estimate_weights_mean(samples, mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, noise, max_weight)
        
        self.assertTrue(w.shape == (3,))
        
        w1 = 2 / (2 - 0.01) * 0.25938780641311815 / 0.28280268445974105
        w2 = 2 / (2 - 0.02) * 0.26653275830062345 / 0.28351593322042085
        w3 = 2 / (2 - 0.03) * 0.22888775934329675 / 0.28423460594051542
        
        self.assertAlmostEqual(w[0], w1)
        self.assertAlmostEqual(w[1], w2)
        self.assertAlmostEqual(w[2], w3)
       
    def test_weight_heuristic(self):
        
        samples = np.array([0.0, 1.0, 2.0])
        mu_gp_t = np.array([0.5, 0.8, 1.2])
        std_gp_t = np.sqrt(np.array([0.1, 0.2, 0.3]))
        mu_gp_s = np.array([0.0, 1.0, 2.0])
        std_gp_s = np.sqrt(np.array([0.01, 0.02, 0.03]))
        noise = 2.0
        max_weight = 1000
        
        w = estimate_weights_heuristic(samples, mu_gp_t, std_gp_t, mu_gp_s, std_gp_s, noise, max_weight)
        
        self.assertTrue(w.shape == (3,))
        
        w1 = 0.25938780641311815 / 0.28139218846178216
        w2 = 0.26653275830062345 / 0.28069480897955168
        w3 = 0.22888775934329675 / 0.28000258891475133
        
        self.assertAlmostEqual(w[0], w1)
        self.assertAlmostEqual(w[1], w2)
        self.assertAlmostEqual(w[2], w3) 