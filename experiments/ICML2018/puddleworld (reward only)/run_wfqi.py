from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.data import load_object
from trlib.algorithms.transfer.wfqi import WFQI, estimate_weights_mean
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from trlib.environments.puddleworld import PuddleWorld
from trlib.utilities.wfqi_utils import estimate_ideal_weights
from trlib.policies.policy import Uniform

""" --- ENVIRONMENTS --- """
target_mdp = PuddleWorld(goal_x=5,goal_y=10, puddle_means=[(1.0,4.0),(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], 
                               puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = False)

actions = [0, 1, 2, 3]
source_data = [load_object("source_data_" + str(i)) for i in [1,2,3]]

""" --- PARAMS --- """

uniform_policy = Uniform(actions)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':2,
                    'min_samples_leaf': 1}

initial_states = [np.array([0.,0.]),np.array([2.5,0.]),np.array([5.,0.]),np.array([7.5,0.]),np.array([10.,0.])]

callback_list = []
#callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_disc", criterion = 'discounted', initial_states = [np.array([0.,0.]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = initial_states))

pre_callback_list = []
#pre_callback_list.append(get_callback_list_entry("eval_policy_pre_callback", policy = uniform_policy, field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = [np.array([0.,0.]) for _ in range(5)]))

fit_params = {}

max_iterations = 60
batch_size = 10
n_steps = 6
n_runs = 20
n_jobs = 5

""" --- WEIGHTS --- """

var_rw = 0.1
var_st = 0.2

""" --- WFQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

kernel_rw = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level = 1.0, noise_level_bounds=(0.01, 1.0))
kernel_st = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level = 1.0, noise_level_bounds=(0.01, 1.0))

algorithm = WFQI(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data, var_rw = var_rw, var_st = var_st, max_gp = 4000,
                 weight_estimator = estimate_weights_mean, max_weight = 1000, kernel_rw = kernel_rw, kernel_st = kernel_st, weight_rw = True, weight_st = [False, False],
                 subtract_noise_rw = True, subtract_noise_st = True, wr = None, ws = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("WFQI-mean", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list)
result = experiment.run(n_jobs)
result.save_json("wfqi-mean.json")