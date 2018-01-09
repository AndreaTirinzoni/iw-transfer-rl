from trlib.utilities.wfqi_utils import generate_source
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from trlib.environments.acrobot import Acrobot
import numpy as np

source_mdp_1 = Acrobot(m1 = 0.8, m2 = 0.8, l1 = 0.8, l2 = 0.8)
source_mdp_2 = Acrobot(m1 = 0.8, m2 = 1.3, l1 = 0.9, l2 = 1.2)
source_mdp_3 = Acrobot(m1 = 0.9, m2 = 0.6, l1 = 1.0, l2 = 0.5)

mdp = source_mdp_3
file_name = "source_data_3"
policy_file = "source_policy_3"

k1 = ConstantKernel(2.45**2, constant_value_bounds="fixed") * RBF(length_scale=[0.488, 0.274, 3.57, 3.77, 2.14], length_scale_bounds="fixed")
k2 = ConstantKernel(1.99**2, constant_value_bounds="fixed") * RBF(length_scale=[3.92, 0.507, 4.27, 1.22, 0.39], length_scale_bounds="fixed")
k3 = ConstantKernel(6.08**2, constant_value_bounds="fixed") * RBF(length_scale=[2.02, 0.662, 0.998, 3.25, 0.000167], length_scale_bounds="fixed")
k4 = ConstantKernel(10.4**2, constant_value_bounds="fixed") * RBF(length_scale=[2.96, 0.324, 2.08, 1.18, 1.82], length_scale_bounds="fixed")
kernel_st =  [k1,k2,k3,k4]

generate_source(mdp, n_episodes = 20, test_fraction = 0, file_name = file_name, policy = None,
                policy_file_name = policy_file, kernel_rw = None, kernel_st = kernel_st, 
                load_data = False, fit_rw = False, fit_st = True, subtract_noise_rw=False, subtract_noise_st=False)