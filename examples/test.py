from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.results import Result
from trlib.experiments.visualizer import plot_steps, plot_experiment
from trlib.algorithms.callbacks import save_json_callback, eval_policy_callback,\
    get_callback_list_entry, get_callbacks
import numpy as np
from trlib.experiments.experiment import Experiment

result_file = "results.json"

mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)
actions = [0, 3, 5, 7, 10, 15, 20, 30]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

fqi = FQI(mdp, pi, verbose = True, actions = actions, batch_size = 5, max_iterations = 30, regressor_type = ExtraTreesRegressor, **regressor_params)

fit_params = {}

callbacks = [eval_policy_callback("perf_disc", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]),
             eval_policy_callback("perf_avg", criterion = 'average', initial_states = [np.array([100.0,1]) for _ in range(5)])]

callback_list = []
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_disc", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_avg", criterion = 'average', initial_states = [np.array([100.0,1]) for _ in range(5)]))

experiment = Experiment("FQI Experiment", fqi, n_steps = 10, n_runs = 4, callback_list = callback_list, **fit_params)
result = experiment.run(4)
result.save_json(result_file)

result = Result.load_json(result_file)
plot_experiment(result, x_name="n_episodes", y_name="perf_disc")
plot_experiment(result, x_name="n_episodes", y_name="perf_avg")
