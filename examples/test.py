from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.results import Result
from trlib.experiments.visualizer import plot_performance
from trlib.algorithms.callbacks import save_json_callback, eval_policy_callback
import numpy as np

result_file = "results.json"

mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)
actions = [0, 3, 5, 7, 10, 15, 20, 30]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

fqi = FQI(mdp, pi, verbose = True, actions = actions, batch_size = 2, max_iterations = 30, regressor_type = ExtraTreesRegressor, **regressor_params)

fit_params = {}

callbacks = [save_json_callback(result_file), 
             eval_policy_callback("perf_disc", criterion = 'discounted', initial_states = [np.array([100.0,1]) for _ in range(5)]),
             eval_policy_callback("perf_avg", criterion = 'average', initial_states = [np.array([100.0,1]) for _ in range(5)])]

fqi.run(3, callbacks, **fit_params)

#result = Result.load_json(result_file)
#plot_performance(result)