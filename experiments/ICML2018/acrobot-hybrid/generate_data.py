from trlib.utilities.wfqi_utils import generate_source
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from trlib.policies.policy import Uniform
from trlib.utilities.data import load_object
from trlib.utilities.evaluation import evaluate_policy
from trlib.environments.acrobot_hybrid import AcrobotHybrid

target_mdp = AcrobotHybrid(m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0)
source_mdp_1 = AcrobotHybrid(l1 = 0.8, l2 = 0.6, m1 = 0.9, m2 = 1.)
source_mdp_2 = AcrobotHybrid(l1 = 0.95, l2 = 0.95, m1 = 0.95, m2 = 1.)
source_mdp_3 = AcrobotHybrid(l1 = 0.85, l2 = 0.85, m1 = 0.9, m2 = 0.9)

mdp = source_mdp_3
file_name = "source_data_3"
policy_file = "source_policy_3"

generate_source(mdp, n_episodes = 200, test_fraction = 0, file_name = file_name, policy = None,
                policy_file_name = policy_file, kernel_rw = None, kernel_st = None, 
                load_data = False, fit_rw = False, fit_st = False, subtract_noise_rw=False, subtract_noise_st=False)