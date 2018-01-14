import numpy as np
from trlib.environments.dam import Dam
from trlib.utilities.evaluation import evaluate_policy
from experiments.ICML2018.dam.encoded_policies import DamPolicyS1, DamPolicyS2,\
    DamPolicyS3

source_mdp_1 = Dam(inflow_profile = 2, alpha = 0.8, beta = 0.2)
source_mdp_2 = Dam(inflow_profile = 3, alpha = 0.35, beta = 0.65)
source_mdp_3 = Dam(inflow_profile = 4, alpha = 0.7, beta = 0.3)
target_mdp = Dam(inflow_profile = 1, alpha = 0.3, beta = 0.7)

policy = DamPolicyS3(0)

initial_states = [np.array([200.0,1]) for _ in range(10)]

print(evaluate_policy(source_mdp_3, policy, criterion = 'discounted', initial_states = initial_states, plt = True))