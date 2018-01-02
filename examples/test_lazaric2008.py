from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.interaction import generate_episodes
from trlib.policies.policy import Uniform
from trlib.algorithms.transfer.lazaric2008 import Lazaric2008

result_file = "lazaric.json"

source_mdp_1 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
source_mdp_4 = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
target_mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)

source_mdps = [source_mdp_1, source_mdp_2, source_mdp_3, source_mdp_4]

actions = [0, 3, 5, 7, 10, 15, 20, 30]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

source_data = [generate_episodes(mdp, Uniform(actions), 10) for mdp in source_mdps]

algorithm = Lazaric2008(target_mdp, pi, actions, batch_size = 5, max_iterations = 60, regressor_type = ExtraTreesRegressor, source_datasets = source_data,
                 delta_sa = 0.1, delta_s_prime = 0.1, delta_r = 0.1, mu = 0.8, n_sample_total = 20000, prior = None, verbose = True, **regressor_params)

fit_params = {}

callback_list = []
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_disc", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_avg", criterion = 'average', initial_states = [np.array([100.0,1]) for _ in range(5)]))

experiment = RepeatExperiment("Lazaric Experiment", algorithm, n_steps = 10, n_runs = 4, callback_list = callback_list, **fit_params)
result = experiment.run(1)
result.save_json(result_file)

result = Result.load_json(result_file)
plot_average([result], "n_episodes", "perf_disc_mean", names = ["Lazaric"], file_name = "plot")