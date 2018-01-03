from joblib import Parallel, delayed
from trlib.experiments.results import ExperimentResult
from trlib.algorithms.callbacks import get_callbacks
import numpy as np

class Experiment:
    """
    Base class for all experiments. An experiment consists of running multiple algorithms multiple times.
    """
    
    def __init__(self, name):
        
        self._name = name
        
class RepeatExperiment(Experiment):
    """
    A repeat-experiment is an experiment where an algorithm is run multiple times.
    
    Notice that the list of callbacks for the algorithm provided to this class must be in the
    form of a callback-specification list (see algorithms.callbacks.py). A list of callback
    functions is not accepted since it is not possible to pickle closures.
    """
    
    def __init__(self, name, algorithm, n_steps, n_runs = 1, callback_list = [], pre_callback_list = [], **algorithm_params):
        
        super().__init__(name)
        
        self._algorithm = algorithm
        self._n_steps = n_steps
        self._n_runs = n_runs
        self._callback_list = callback_list
        self._pre_callback_list = pre_callback_list
        self._algorithm_params = algorithm_params
        
        self._result = ExperimentResult(name, n_runs = n_runs)
        
    def _run_algorithm(self, seed = None):
        
        if seed is not None:
            np.random.seed(seed)
        self._algorithm.reset()
        callbacks = get_callbacks(self._callback_list)
        pre_callbacks = get_callbacks(self._pre_callback_list)
        return self._algorithm.run(self._n_steps, callbacks, pre_callbacks, **self._algorithm_params)
    
    def run(self, n_jobs = 1):
        """
        Runs the experiment over n_jobs processes.
        
        Parameters
        ----------
        n_jobs: number of processes to run
        """
        
        if n_jobs == 1:
            results = [self._run_algorithm() for _ in range(self._n_runs)]
        elif n_jobs > 1:
            seeds = [np.random.randint(1000000) for _ in range(self._n_runs)]
            results = Parallel(n_jobs = n_jobs)(delayed(self._run_algorithm)(seed) for seed in seeds)
        
        for result in results:
            self._result.add_run(result)
            
        return self._result
        