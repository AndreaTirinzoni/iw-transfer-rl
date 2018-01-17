from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.data import load_object
from trlib.algorithms.transfer.wfqi import WFQI
from trlib.utilities.wfqi_utils import estimate_ideal_weights

""" --- ENVIRONMENTS --- """
source_mdp_1 = Dam(inflow_profile = 2, alpha = 0.8, beta = 0.2)
source_mdp_2 = Dam(inflow_profile = 3, alpha = 0.35, beta = 0.65)
source_mdp_3 = Dam(inflow_profile = 4, alpha = 0.7, beta = 0.3)
source_mdp_4 = Dam(inflow_profile = 5, alpha = 0.4, beta = 0.6)
source_mdp_5 = Dam(inflow_profile = 6, alpha = 0.6, beta = 0.4)
source_mdp_6 = Dam(inflow_profile = 7, alpha = 0.45, beta = 0.55)
target_mdp = Dam(inflow_profile = 1, alpha = 0.3, beta = 0.7)
source_mdps = [source_mdp_1,source_mdp_2,source_mdp_3,source_mdp_4,source_mdp_5,source_mdp_6]

actions = [0, 3, 5, 7, 10, 15, 20, 30]
source_data = [load_object("source_data_" + str(i)) for i in [1,2,3,4,5,6]]

""" --- PARAMS --- """

regressor_params = {'n_estimators': 100,
                    'criterion': 'mse',
                    'min_samples_split':20}

initial_states = [np.array([200.0,1]) for _ in range(10)]

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = initial_states))
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_avg_greedy", criterion = 'average', initial_states = initial_states))

pre_callback_list = []

fit_params = {}

max_iterations = 60
batch_size = 1
n_steps = 10
n_runs = 20
n_jobs = 10

""" --- WEIGHTS --- """

var_rw = 0.5 ** 2
var_st = 2.0 ** 2
wr,ws = estimate_ideal_weights(target_mdp, source_mdps, [data[0] for data in source_data], var_rw, var_st)

""" --- WFQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = WFQI(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data, var_rw = var_rw, var_st = var_st, max_gp = 0,
                 weight_estimator = None, max_weight = 1000, kernel_rw = None, kernel_st = None, weight_rw = None, weight_st = None,
                 subtract_noise_rw = False, subtract_noise_st = False, wr = wr, ws = ws, verbose = True, **regressor_params)

experiment = RepeatExperiment("WFQI-ideal", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list)
result = experiment.run(n_jobs)
result.save_json("wfqi-ideal.json")