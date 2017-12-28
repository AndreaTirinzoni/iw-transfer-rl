from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.results import save_json_callback, load_json
from trlib.experiments.visualizer import plot_performance

result_file = "results.json"

result = load_json(result_file)
plot_performance(result)

mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)
actions = [0, 3, 5, 7, 10, 15, 20, 30]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

fqi = FQI(mdp, pi, max_steps = 10, verbose = True, actions = actions, batch_size = 2, max_iterations = 50, regressor_type = ExtraTreesRegressor, **regressor_params)

fit_params = {}

callbacks = [save_json_callback(result_file)]

fqi.run(callbacks, **fit_params)

result = load_json(result_file)
plot_performance(result)