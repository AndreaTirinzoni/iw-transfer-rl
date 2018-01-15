from trlib.environments.dam import Dam
from experiments.ICML2018.dam.encoded_policies import DamPolicyT, DamPolicyS1,\
    DamPolicyS2, DamPolicyS3
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from trlib.utilities.wfqi_utils import generate_source

source_mdp_1 = Dam(inflow_profile = 2, alpha = 0.8, beta = 0.2)
source_mdp_2 = Dam(inflow_profile = 3, alpha = 0.35, beta = 0.65)
source_mdp_3 = Dam(inflow_profile = 4, alpha = 0.7, beta = 0.3)
target_mdp = Dam(inflow_profile = 1, alpha = 0.3, beta = 0.7)

mdp = source_mdp_3
file_name = "source_data_3"
policy = DamPolicyS3(0.1)

kernel_rw = ConstantKernel(11.9**2, constant_value_bounds = "fixed") * RBF(length_scale=[1.58, 1e+05, 0.000567], length_scale_bounds = "fixed")
kernel_st = ConstantKernel(213**2, constant_value_bounds = "fixed") * RBF(length_scale = 215, length_scale_bounds = "fixed") + WhiteKernel(noise_level = 4.0, noise_level_bounds = "fixed")
kernel_st = [kernel_st]

generate_source(mdp, n_episodes = 20, test_fraction = 0, file_name = file_name, policy = policy,
                policy_file_name = None, kernel_rw = kernel_rw, kernel_st = kernel_st, 
                load_data = False, fit_rw = True, fit_st = True, subtract_noise_rw=False, subtract_noise_st=True)