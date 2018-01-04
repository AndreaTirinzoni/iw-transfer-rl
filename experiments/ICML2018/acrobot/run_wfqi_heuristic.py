from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.data import load_object
from trlib.algorithms.transfer.wfqi import WFQI, estimate_weights_heuristic
from sklearn.gaussian_process.kernels import RBF
from trlib.policies.policy import Uniform
from trlib.environments.acrobot import Acrobot

""" --- ENVIRONMENTS --- """
target_mdp = Acrobot(m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0)

actions = [0, 1]
source_data = [load_object("source_data_" + str(i)) for i in [1,2,3]]

""" --- PARAMS --- """

uniform_policy = Uniform(actions)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':5,
                    'min_samples_leaf': 2}

initial_states = [np.array([-2.0,0.,0.,0.]),np.array([-1.5,0.,0.,0.]),np.array([-1.0,0.,0.,0.]),
                  np.array([-0.5,0.,0.,0.]),np.array([0.0,0.,0.,0.]),np.array([0.5,0.,0.,0.]),
                  np.array([1.0,0.,0.,0.]),np.array([1.5,0.,0.,0.]),np.array([2.0,0.,0.,0.])]

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = initial_states))

pre_callback_list = []

fit_params = {}

max_iterations = 100
batch_size = 5
n_steps = 10
n_runs = 20
n_jobs = 5

""" --- WEIGHTS --- """

var_st = 0.5

""" --- WFQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.1)

kernel_st =  1.0 * RBF(length_scale=np.array([1.0,1.0,1.0,1.0,1.0]), length_scale_bounds=(0.01,10.0))

algorithm = WFQI(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data, var_rw = 0, var_st = var_st, max_gp = 5000,
                 weight_estimator = estimate_weights_heuristic, max_weight = 1000, kernel_rw = None, kernel_st = kernel_st, weight_rw = False, weight_st = [True, True, True, True],
                 subtract_noise_rw = False, subtract_noise_st = False, wr = None, ws = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("WFQI-heuristic", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list)
result = experiment.run(n_jobs)
result.save_json("wfqi-heuristic.json")