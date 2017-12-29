from joblib import Parallel, delayed
from trlib.experiments.results import ExperimentResult

class Experiment:
    
    def __init__(self, name, algorithm, n_steps, n_runs = 1, callbacks = [], **algorithm_params):
        
        self._name = name
        self._algorithm = algorithm
        self._n_steps = n_steps
        self._n_runs = n_runs
        self._callbacks = callbacks
        self._algorithm_params = algorithm_params
        
        self._result = ExperimentResult(name, n_runs = n_runs)
        
    def _run_algorithm(self):
        
        self._algorithm.reset()
        return self._algorithm.run(self._n_steps, self._callbacks, **self._algorithm_params)
    
    def run(self, n_jobs = 1):
        
        if n_jobs == 1:
            results = [self._run_algorithm() for _ in range(self._n_runs)]
        elif n_jobs > 1:
            results = Parallel(n_jobs = n_jobs)(delayed(self._run_algorithm)() for _ in range(self._n_runs))
        
        for result in results:
            self._result.add_run(result)
            
        return self._result
        