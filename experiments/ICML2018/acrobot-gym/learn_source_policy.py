from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.experiments.visualization import plot_average
from trlib.algorithms.callbacks import  get_callback_list_entry
import numpy as np
from trlib.experiments.experiment import RepeatExperiment
from trlib.utilities.data import save_object
from trlib.utilities.evaluation import evaluate_policy
from trlib.environments.acrobot_gym import AcrobotGym

source_mdp_1 = AcrobotGym(m1 = 0.8, m2 = 1.3, l1 = 0.7, l2 = 1.2) # -112 on target (-145 from vertical)
source_mdp_2 = AcrobotGym(m1 = 1.2, m2 = 0.6, l1 = 1.1, l2 = 0.5) # -110 on target (-319 from vertical)
target_mdp = AcrobotGym(m1 = 1.0, m2 = 1.0, l1 = 1.0, l2 = 1.0) # -70 (-110 from vertical)

mdp = source_mdp_1
file_name = "source_policy_1"

actions = [0, 1, 2]
pi = EpsilonGreedy(actions, ZeroQ(), 0.1)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':5,
                    'min_samples_leaf': 2}

fqi = FQI(mdp, pi, verbose = True, actions = actions, batch_size = 50, max_iterations = 100, regressor_type = ExtraTreesRegressor, **regressor_params)

callback_list = []
callback_list.append(get_callback_list_entry("eval_greedy_policy_callback", field_name = "perf_disc_greedy", criterion = 'discounted', n_episodes = 5))

experiment = RepeatExperiment("FQI Experiment", fqi, n_steps = 5, n_runs = 1, callback_list = callback_list)
result = experiment.run(1)

plot_average([result], "n_episodes", "perf_disc_greedy_mean", names = ["FQI"])
plot_average([result], "n_episodes", "n_samples", names = ["FQI"])

policy = EpsilonGreedy(actions, pi.Q, 0)

print(evaluate_policy(mdp, policy, criterion = 'discounted', n_episodes = 5))

save_object(policy, file_name)