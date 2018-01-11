from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.data import load_object
from trlib.algorithms.transfer.wfqi import WFQI, estimate_weights_mean
from sklearn.gaussian_process.kernels import RBF
from trlib.policies.policy import Uniform
from trlib.environments.acrobot_gym import AcrobotGym

""" --- ENVIRONMENTS --- """
target_mdp = AcrobotGym(m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0)

actions = [0, 1, 2]
source_data = [load_object("source_data_" + str(i)) for i in [1,2]]

""" --- PARAMS --- """

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':5,
                    'min_samples_leaf': 2}

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', n_episodes = 5))

pre_callback_list = []

fit_params = {}

max_iterations = 100
batch_size = 50
n_steps = 5
n_runs = 10
n_jobs = 10

""" --- WEIGHTS --- """

var_st = 0.1

""" --- WFQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.1)

k1 = ConstantKernel(1.59**2, constant_value_bounds="fixed") * RBF(length_scale=1.86, length_scale_bounds="fixed")
k2 = ConstantKernel(2.09**2, constant_value_bounds="fixed") * RBF(length_scale=0.828, length_scale_bounds="fixed")
k3 = ConstantKernel(2.5**2, constant_value_bounds="fixed") * RBF(length_scale=2.89, length_scale_bounds="fixed")
k4 = ConstantKernel(3.14**2, constant_value_bounds="fixed") * RBF(length_scale=2.76, length_scale_bounds="fixed")
kernel_st = [k1,k2,k3,k4]

algorithm = WFQI(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data, var_rw = 0, var_st = var_st, max_gp = 10000,
                 weight_estimator = estimate_weights_mean, max_weight = 1000, kernel_rw = None, kernel_st = kernel_st, weight_rw = False, weight_st = [True, True, True, True],
                 subtract_noise_rw = False, subtract_noise_st = False, wr = None, ws = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("WFQI-mean", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list)
result = experiment.run(n_jobs)
result.save_json("wfqi-mean.json")