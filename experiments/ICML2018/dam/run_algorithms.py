from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.algorithms.transfer.lazaric2008 import Lazaric2008
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.algorithms.transfer.laroche2017 import Laroche2017
from trlib.utilities.data import load_object
from trlib.environments.dam import Dam

""" --- ENVIRONMENTS --- """
target_mdp = Dam(inflow_profile = 1, alpha = 0.3, beta = 0.7)

actions = [0, 3, 5, 7, 10, 15, 20, 30]
source_data = [load_object("source_data_" + str(i))[0] for i in [1,2,3,4,5,6]]

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

""" --- FQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = FQI(target_mdp, pi, verbose = True, actions = actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, **regressor_params)

experiment = RepeatExperiment("FQI", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list, pre_callback_list = pre_callback_list, **fit_params)
result = experiment.run(n_jobs)
result.save_json("fqi.json")

""" --- LAZARIC --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = Lazaric2008(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data,
                 delta_sa = 0.1, delta_s_prime = 0.1, delta_r = 0.1, mu = 0.8, n_sample_total = 60000, prior = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("lazaric2008", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list, pre_callback_list = pre_callback_list, **fit_params)
result = experiment.run(n_jobs)
result.save_json("lazaric2008.json")
