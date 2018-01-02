from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.interaction import generate_episodes
from trlib.policies.policy import Uniform
from trlib.algorithms.transfer.lazaric2008 import Lazaric2008
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.algorithms.transfer.laroche2017 import Laroche2017

""" --- ENVIRONMENTS --- """
source_mdp_1 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
source_mdp_4 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
target_mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)

source_mdps = [source_mdp_1, source_mdp_2, source_mdp_3, source_mdp_4]

actions = [0, 3, 5, 7, 10, 15, 20, 30]
source_data = [generate_episodes(mdp, Uniform(actions), 30) for mdp in source_mdps]

""" --- PARAMS --- """

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

callback_list = []
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_disc", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_avg", criterion = 'average', initial_states = [np.array([100.0,1]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]))

fit_params = {}

""" --- FQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = FQI(target_mdp, pi, verbose = True, actions = actions, batch_size = 5, max_iterations = 50, regressor_type = ExtraTreesRegressor, **regressor_params)

experiment = RepeatExperiment("FQI", algorithm, n_steps = 10, n_runs = 20, callback_list = callback_list, **fit_params)
result = experiment.run(5)
result.save_json("fqi.json")

""" --- LAROCHE --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = Laroche2017(target_mdp, pi, verbose = True, actions = actions, batch_size = 5, max_iterations = 50, regressor_type = ExtraTreesRegressor, source_datasets=source_data, **regressor_params)

experiment = RepeatExperiment("laroche2017", algorithm, n_steps = 10, n_runs = 20, callback_list = callback_list, **fit_params)
result = experiment.run(5)
result.save_json("laroche2017.json")

""" --- LAZARIC --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = Lazaric2008(target_mdp, pi, actions, batch_size = 5, max_iterations = 50, regressor_type = ExtraTreesRegressor, source_datasets = source_data,
                 delta_sa = 0.1, delta_s_prime = 0.1, delta_r = 0.1, mu = 0.8, n_sample_total = 43200, prior = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("lazaric2008", algorithm, n_steps = 10, n_runs = 20, callback_list = callback_list, **fit_params)
result = experiment.run(5)
result.save_json("lazaric2008.json")


