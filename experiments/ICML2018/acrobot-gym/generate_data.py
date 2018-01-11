from trlib.utilities.wfqi_utils import generate_source
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from trlib.environments.acrobot_gym import AcrobotGym
from trlib.policies.policy import Uniform
from trlib.utilities.data import load_object
from trlib.utilities.evaluation import evaluate_policy

source_mdp_1 = AcrobotGym(m1 = 0.8, m2 = 1.3, l1 = 0.7, l2 = 1.2)
source_mdp_2 = AcrobotGym(m1 = 1.2, m2 = 0.6, l1 = 1.1, l2 = 0.5)
target_mdp = AcrobotGym(m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0)

mdp = source_mdp_2
file_name = "source_data_2"
policy_file = "source_policy_2"

k1 = ConstantKernel(1.59**2, constant_value_bounds="fixed") * RBF(length_scale=1.86, length_scale_bounds="fixed")
k2 = ConstantKernel(2.09**2, constant_value_bounds="fixed") * RBF(length_scale=0.828, length_scale_bounds="fixed")
k3 = ConstantKernel(2.5**2, constant_value_bounds="fixed") * RBF(length_scale=2.89, length_scale_bounds="fixed")
k4 = ConstantKernel(3.14**2, constant_value_bounds="fixed") * RBF(length_scale=2.76, length_scale_bounds="fixed")
kernel_st = [k1,k2,k3,k4]

generate_source(mdp, n_episodes = 50, test_fraction = 0, file_name = file_name, policy = None,
                policy_file_name = policy_file, kernel_rw = None, kernel_st = kernel_st, 
                load_data = False, fit_rw = False, fit_st = True, subtract_noise_rw=False, subtract_noise_st=False)