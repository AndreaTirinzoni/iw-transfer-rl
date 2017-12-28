import numpy as np
import matplotlib.pyplot as plt
from trlib.environments.dam import Dam
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble.forest import ExtraTreesRegressor

mdp = Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)
actions = [0, 3, 5, 7, 10, 15, 20, 30]
pi = EpsilonGreedy(actions, ZeroQ(), 0.3)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2}

fqi = FQI(mdp, pi, n_episodes = 100, verbose = True, actions = actions, batch_size = 5, max_iterations = 50, regressor_type = ExtraTreesRegressor, **regressor_params)

fit_params = {}

nruns = 10
N = np.zeros(nruns)
J = np.zeros(nruns)

for i in range(nruns):
    N[i],J[i] = fqi.step(**fit_params)

plt.plot(N,J)
plt.show()

