from joblib import Parallel, delayed
from trlib.experiments.results import ExperimentResult
from trlib.algorithms.callbacks import get_callbacks
import numpy as np

class Experiment:
    
    def __init__(self, name, algorithm, n_steps, n_runs = 1, callback_list = [], **algorithm_params):
        
        self._name = name
        self._algorithm = algorithm
        self._n_steps = n_steps
        self._n_runs = n_runs
        self._callback_list = callback_list
        self._algorithm_params = algorithm_params
        
        self._result = ExperimentResult(name, n_runs = n_runs)
        
    def _run_algorithm(self, seed = None):
        
        if seed is not None:
            np.random.seed(seed)
        self._algorithm.reset()
        callbacks = get_callbacks(self._callback_list)
        return self._algorithm.run(self._n_steps, callbacks, **self._algorithm_params)
    
    def run(self, n_jobs = 1):
        
        if n_jobs == 1:
            results = [self._run_algorithm() for _ in range(self._n_runs)]
        elif n_jobs > 1:
            seeds = [np.random.randint(1000000) for _ in range(self._n_runs)]
            results = Parallel(n_jobs = n_jobs)(delayed(self._run_algorithm)(seed) for seed in seeds)
        
        for result in results:
            self._result.add_run(result)
            
        return self._result
        