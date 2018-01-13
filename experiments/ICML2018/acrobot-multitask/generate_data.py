from trlib.utilities.wfqi_utils import generate_source
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared
import numpy as np
from trlib.utilities.data import load_object
from trlib.utilities.evaluation import evaluate_policy
from acro_policy import AcrobotPolicy
from trlib.environments.acrobot_multitask import AcrobotMultitask

source_mdp_1 = AcrobotMultitask(m1 = 0.9, m2 = 0.6, l1 = 1.1, l2 = 0.7, task = "swing-up")
source_mdp_2 = AcrobotMultitask(l1 = 0.95, l2 = 0.95, m1 = 0.95, m2 = 1.0, task = "rotate")

mdp = source_mdp_2
file_name = "source_data_2"
policy_file = "source_policy_2"

k1 = ConstantKernel(2.74**2, constant_value_bounds="fixed") * RBF(length_scale=1.51, length_scale_bounds="fixed")
k2 = ConstantKernel(2.14**2, constant_value_bounds="fixed") * RBF(length_scale=0.92, length_scale_bounds="fixed")
k3 = ConstantKernel(2.42**2, constant_value_bounds="fixed") * RBF(length_scale=2.47, length_scale_bounds="fixed")
k4 = ConstantKernel(3.14**2, constant_value_bounds="fixed") * RBF(length_scale=2.76, length_scale_bounds="fixed")
kernel_st = [k1,k2,k3,k4]

kernel_rw = ConstantKernel(2.03**2, constant_value_bounds="fixed")  * RBF(length_scale=2.57, length_scale_bounds="fixed")

generate_source(mdp, n_episodes = 50, test_fraction = 0, file_name = file_name, policy = None,
                policy_file_name = policy_file, kernel_rw = None, kernel_st = None, 
                load_data = False, fit_rw = True, fit_st = True, subtract_noise_rw=False, subtract_noise_st=False)