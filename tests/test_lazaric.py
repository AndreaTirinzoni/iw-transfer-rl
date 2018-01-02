import numpy as np
from unittest import TestCase
from trlib.algorithms.transfer.lazaric2008 import _distance, _phi, _weights,\
    _avg_distances, _relevance, _compliance, _compliance_relevance

class TestLazaric(TestCase):
    
    def test_distance(self):
        
        x = np.array([2,4])
        y = np.array([5,8])
        
        self.assertTrue(np.all(np.isclose(np.array([3,4]), _distance(x, y))))
        
        x = np.array([[2,4],[2,1]])
        y = np.array([[5,8],[1,2]])
        
        self.assertTrue(np.all(np.isclose(np.array([5,np.sqrt(2)]), _distance(x, y))))
        
    def test_phi(self):
        
        self.assertAlmostEqual(_phi(0,0.1),1)
        self.assertAlmostEqual(_phi(1,0.1),np.exp(-10))
        self.assertAlmostEqual(_phi(2,0.1),np.exp(-40))
        
        x = np.array([0,1,2])
        self.assertTrue(np.all(np.isclose(np.array([1,np.exp(-0.1),np.exp(-0.4)]),_phi(x,10))))
        
    def test_weights(self):
        
        source_sa = np.array([[0,1],[1,2],[2,3]])
        target_sa = np.array([[1,0],[0,1]])
        delta_sa = 1
        
        w, dist = _weights(target_sa, source_sa, delta_sa)
        
        self.assertEqual((2,3), w.shape)
        self.assertEqual((2,3), dist.shape)
        
        d = np.array([[np.sqrt(2),2,np.sqrt(10)],[0,np.sqrt(2),np.sqrt(8)]])
        w2 = np.exp(-d**2)
        w2 = w2 / np.sum(w2,1)[:,np.newaxis]
        self.assertTrue(np.all(np.isclose(d , dist)))
        self.assertTrue(np.all(np.isclose(w2 , w)))
        
        source_sa = np.array([[0,1],[1,2]])
        target_sa = np.array([[1,0],[1,1]])
        delta_sa = 1
        
        w, dist = _weights(target_sa, source_sa, delta_sa)
        
        self.assertEqual((2,2), w.shape)
        self.assertEqual((2,2), dist.shape)
        
        d = np.array([[np.sqrt(2),2],[1,1]])
        w2 = np.exp(-d**2)
        w2 = w2 / np.sum(w2,1)[:,np.newaxis]
        self.assertTrue(np.all(np.isclose(d , dist)))
        self.assertTrue(np.all(np.isclose(w2 , w)))

    def test_avg_distances(self):
        
        w = np.array([[0.3,0.1,0.6],[0.4,0.2,0.4],[0.1,0.7,0.2]])
        d = np.array([[1,2,3],[4,5,6],[7,8,9]])
        mu = 0.5
        
        self.assertTrue(np.all(np.isclose(np.array([1.5,5,8]), _avg_distances(w, d, mu))))
 
    def test_relevance(self):
        
        lambdas = np.array([1,2,3])
        avg_dist = np.array([4,5,6])
        
        self.assertTrue(np.all(np.isclose(np.array([np.exp(- (5.0/24) ** 2), np.exp(- (2.0/15) ** 2), np.exp(- (1.0/12) ** 2)]), _relevance(lambdas,avg_dist))))    
    
    def test_compliance(self):
        
        s_s = np.array([1,2,3])
        s_s_prime = np.array([2,3,5])
        s_r = np.array([7,8,9])
        t_s = np.array([0,1])
        t_s_prime = np.array([1,2])
        t_r = np.array([6,7])
        
        delta_r = 0.1
        delta_s_prime = 0.1
        
        w = np.array([[1,2,3],[4,5,6]])
        
        lp11 = _phi(abs(1 - 1), delta_s_prime)
        lp12 = _phi(abs(1 - 1), delta_s_prime)
        lp13 = _phi(abs(1 - 2), delta_s_prime)
        lp21 = _phi(abs(2 - 2), delta_s_prime)
        lp22 = _phi(abs(2 - 2), delta_s_prime)
        lp23 = _phi(abs(2 - 3), delta_s_prime)
        lp = np.array([[lp11,lp12,lp13],[lp21,lp22,lp23]]) * w
        
        lr11 = _phi(abs(6 - 7), delta_r)
        lr12 = _phi(abs(6 - 8), delta_r)
        lr13 = _phi(abs(6 - 9), delta_r)
        lr21 = _phi(abs(7 - 7), delta_r)
        lr22 = _phi(abs(7 - 8), delta_r)
        lr23 = _phi(abs(7 - 9), delta_r)
        lr = np.array([[lr11,lr12,lr13],[lr21,lr22,lr23]]) * w
        
        comp = _compliance(w, t_s, t_r, t_s_prime, s_s, s_r, s_s_prime, delta_s_prime, delta_r)
        
        comp2 = np.mean(lp,1) * np.mean(lr,1)
        
        self.assertTrue(np.all(np.isclose(comp,comp2)))

    def test_compliance_relevance(self):
        
        s_s = np.array([1,2,3])
        s_sa = np.array([[1,0],[2,1],[3,2]])
        s_s_prime = np.array([2,3,5])
        s_r = np.array([7,8,9])
        t_s = np.array([0,1])
        t_sa = np.array([[0,0],[1,1]])
        t_s_prime = np.array([1,2])
        t_r = np.array([6,7])
        
        delta_sa = 0.5
        delta_r = 0.4
        delta_s_prime = 0.2
        prior = 0.5
        mu = 0.8
        
        comp, rel = _compliance_relevance(t_sa, t_s, t_r, t_s_prime, s_sa, s_s, s_r, s_s_prime, prior, delta_sa, delta_s_prime, delta_r, mu)
        
        w, dist = _weights(t_sa, s_sa, delta_sa)
        comp2 = _compliance(w, t_s, t_r, t_s_prime, s_s, s_r, s_s_prime, delta_s_prime, delta_r)
        comp2 = np.sum(comp2) * prior / 2
        w2, dist2 = _weights(s_sa, t_sa, delta_sa)
        comp3 = _compliance(w2, s_s, s_r, s_s_prime, t_s, t_r, t_s_prime, delta_s_prime, delta_r)
        avg_dist = _avg_distances(w2, dist2, mu)
        
        self.assertTrue(np.all(np.isclose(comp, comp2)))
        self.assertTrue(np.all(np.isclose(rel, _relevance(comp3, avg_dist))))