from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.callbacks import get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.algorithms.transfer.lazaric2008 import Lazaric2008
from trlib.algorithms.reinforcement.fqi import FQI
from trlib.algorithms.transfer.laroche2017 import Laroche2017
from trlib.environments.puddleworld import PuddleWorld
from trlib.utilities.data import load_object
from trlib.policies.policy import Uniform

""" --- ENVIRONMENTS --- """
target_mdp = PuddleWorld(goal_x=5,goal_y=10, puddle_means=[(1.0,4.0),(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], 
                               puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = True)

actions = [0, 1, 2, 3]
source_data = [load_object("source_data_" + str(i))[0] for i in [1,2,3]]

""" --- PARAMS --- """

uniform_policy = Uniform(actions)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':2,
                    'min_samples_leaf': 1}

initial_states = [np.array([0.,0.]) for _ in range(5)]

callback_list = []
#callback_list.append(get_callback_list_entry("eval_policy_callback", field_name = "perf_disc", criterion = 'discounted', initial_states = [np.array([0.,0.]) for _ in range(5)]))
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = initial_states))

pre_callback_list = []
#pre_callback_list.append(get_callback_list_entry("eval_policy_pre_callback", policy = uniform_policy, field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = [np.array([0.,0.]) for _ in range(5)]))

fit_params = {}

max_iterations = 60
batch_size = 10
n_steps = 6
n_runs = 20
n_jobs = 10

""" --- FQI --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = FQI(target_mdp, pi, verbose = True, actions = actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, **regressor_params)

experiment = RepeatExperiment("FQI", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list, pre_callback_list = pre_callback_list, **fit_params)
result = experiment.run(n_jobs)
result.save_json("fqi.json")

""" --- LAROCHE --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = Laroche2017(target_mdp, pi, verbose = True, actions = actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets=source_data, **regressor_params)

experiment = RepeatExperiment("laroche2017", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list, pre_callback_list = pre_callback_list, **fit_params)
result = experiment.run(n_jobs)
result.save_json("laroche2017.json")

""" --- LAZARIC --- """

pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

algorithm = Lazaric2008(target_mdp, pi, actions, batch_size = batch_size, max_iterations = max_iterations, regressor_type = ExtraTreesRegressor, source_datasets = source_data,
                 delta_sa = 0.1, delta_s_prime = 0.1, delta_r = 0.1, mu = 0.8, n_sample_total = 500, prior = None, verbose = True, **regressor_params)

experiment = RepeatExperiment("lazaric2008", algorithm, n_steps = n_steps, n_runs = n_runs, callback_list = callback_list, pre_callback_list = pre_callback_list, **fit_params)
result = experiment.run(n_jobs)
result.save_json("lazaric2008.json")
