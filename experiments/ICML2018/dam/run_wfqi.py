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

""" --- ENVIRONMENTS --- """
target_mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)

actions = [0, 3, 5, 7, 10, 15, 20, 30]
source_data = [load_object("source_data_" + str(i)) for i in [1,2,3]]

""" --- PARAMS --- """

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]))

pre_callback_list = []

fit_params = {}

max_iterations = 60
batch_size = 1
n_steps = 10
n_runs = 5
n_jobs = 5

""" --- WEIGHTS --- """

var_rw = 1.0
var_st = 20.0

""" --- WFQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

kernel_rw = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 1000.0))
kernel_st = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01,1000.0)) + WhiteKernel(noise_level = 10.0, noise_level_bounds=(1.0, 50.0))

algorithm = WFQI(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data, var_rw = var_rw, var_st = var_st, max_gp = 4000,
                 weight_estimator = estimate_weights_mean, max_weight = 1000, kernel_rw = kernel_rw, kernel_st = kernel_st, weight_rw = True, weight_st = [True, False],
                 subtract_noise_rw = False, subtract_noise_st = True, wr = None, ws = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("WFQI-mean", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list)
result = experiment.run(n_jobs)
result.save_json("wfqi-mean.json")