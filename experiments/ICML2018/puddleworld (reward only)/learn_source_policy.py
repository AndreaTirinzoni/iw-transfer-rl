from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average
from trlib.algorithms.callbacks import  get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.environments.puddleworld import PuddleWorld
from trlib.utilities.data import save_object
from trlib.utilities.evaluation import evaluate_policy

source_mdp_1 = PuddleWorld(goal_x=5,goal_y=10, puddle_slow = False)
source_mdp_2 = PuddleWorld(goal_x=5,goal_y=10, puddle_means=[(2.0,2.0),(4.0,6.0),(1.0,8.0), (2.0, 4.0), (8.5,7.0),(8.5,5.0)], 
                                 puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = False)
source_mdp_3 = PuddleWorld(goal_x=7,goal_y=10, puddle_means=[(8.0,2.0), (1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)],
                                 puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = False)
source_mdps = [source_mdp_1,source_mdp_2,source_mdp_3]

target_mdp = PuddleWorld(goal_x=5,goal_y=10, puddle_means=[(1.0,4.0),(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], 
                               puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)], puddle_slow = False)

mdp = target_mdp
file_name = "target_policy"

actions = [0, 1, 2, 3]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':2,
                    'min_samples_leaf': 1}

fqi = FQI(mdp, pi, verbose = True, actions = actions, batch_size = 50, max_iterations = 60, regressor_type = ExtraTreesRegressor, **regressor_params)

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', initial_states = [np.array([0., 0.]) for _ in range(5)]))

experiment = RepeatExperiment("FQI Experiment", fqi, n_steps = 5, n_runs = 1, callback_list = callback_list)
result = experiment.run(1)

plot_average([result], "n_episodes", "perf_disc_greedy_mean", names = ["FQI"])
plot_average([result], "n_episodes", "n_samples", names = ["FQI"])

policy = EpsilonGreedy(actions, pi.Q, 0)

print(evaluate_policy(mdp, policy, criterion = 'discounted', initial_states = [np.array([0., 0.]) for _ in range(5)]))

save_object(policy, file_name)