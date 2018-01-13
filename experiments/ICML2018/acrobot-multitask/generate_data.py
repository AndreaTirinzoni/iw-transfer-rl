from trlib.utilities.wfqi_utils import generate_source
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from trlib.policies.policy import Uniform
from trlib.utilities.data import load_object
from trlib.utilities.evaluation import evaluate_policy
from trlib.environments.acrobot_hybrid import AcrobotHybrid
from acro_policy import AcrobotPolicy

source_mdp_1 = AcrobotHybrid(m1 = 0.8, m2 = 0.8, l1 = 0.8, l2 = 0.8)
source_mdp_2 = AcrobotHybrid(m1 = 0.8, m2 = 1.3, l1 = 0.9, l2 = 1.2)
source_mdp_3 = AcrobotHybrid(m1 = 0.9, m2 = 0.6, l1 = 1.1, l2 = 0.7)

mdp = source_mdp_3
file_name = "source_data_3_200"

k1 = ConstantKernel(2.74**2, constant_value_bounds="fixed") * RBF(length_scale=1.51, length_scale_bounds="fixed")
k2 = ConstantKernel(2.14**2, constant_value_bounds="fixed") * RBF(length_scale=0.92, length_scale_bounds="fixed")
k3 = ConstantKernel(2.42**2, constant_value_bounds="fixed") * RBF(length_scale=2.47, length_scale_bounds="fixed")
k4 = ConstantKernel(3.14**2, constant_value_bounds="fixed") * RBF(length_scale=2.76, length_scale_bounds="fixed")
kernel_st = [k1,k2,k3,k4]

generate_source(mdp, n_episodes = 200, test_fraction = 0, file_name = file_name, policy = AcrobotPolicy(0.1),
                policy_file_name = None, kernel_rw = None, kernel_st = None, 
                load_data = False, fit_rw = False, fit_st = True, subtract_noise_rw=False, subtract_noise_st=False)