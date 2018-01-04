from trlib.environments.puddleworld import PuddleWorld
from trlib.utilities.wfqi_utils import generate_source
from trlib.utilities.data import load_object
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

source_mdp_1 = PuddleWorld(goal_x=5,goal_y=10, puddle_slow = True)
source_mdp_2 = PuddleWorld(goal_x=5,goal_y=10, puddle_means=[(2.0,2.0),(4.0,6.0),(1.0,8.0), (2.0, 4.0), (8.5,7.0),(8.5,5.0)], 
                                 puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = True)
source_mdp_3 = PuddleWorld(goal_x=7,goal_y=10, puddle_means=[(8.0,2.0), (1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)],
                                 puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = True)

mdp = source_mdp_3
file_name = "source_data_3"
policy_file = "source_policy_3"

kernel_rw = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level = 1.0, noise_level_bounds=(0.01, 1.0))
kernel_st = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level = 1.0, noise_level_bounds=(0.01, 1.0))

generate_source(mdp, n_episodes = 20, test_fraction = 0, file_name = file_name, policy = None,
                policy_file_name = policy_file, kernel_rw = kernel_rw, kernel_st = kernel_st, 
                load_data = False, fit_rw = True, fit_st = True, subtract_noise_rw=True, subtract_noise_st=True)