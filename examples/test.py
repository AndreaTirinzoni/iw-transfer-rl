from trlib.policies.qfunction import QFunction
from trlib.policies.valuebased import ValueBased, EpsilonGreedy

Q = QFunction()
actions = [1,2,3]
pi = EpsilonGreedy(actions,Q,0.5)