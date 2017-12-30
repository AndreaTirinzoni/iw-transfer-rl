import json

class Result:
    """
    Base class for all results
    """
    
    def add_fields(self, **kwargs):
        for name,value in kwargs.items():
            setattr(self, name, value) 
    
    @staticmethod
    def load_json(file_name):
        """
        Loads a json file and returns the corresponding Result object
        
        Parameters
        ----------
        file_name: the file to load
        
        Returns
        -------
        The restored result object
        """
        
        with open(file_name,"r") as file:
            s = json.load(file)
            result = Result()
            result.__dict__ = s
            return result
        
    def save_json(self, file_name):
        with open(file_name,"w") as file:
            json.dump(self.__dict__,file)
    

class AlgorithmResult(Result):
    """
    A class to store the results of running an algorithm
    """
    
    def __init__(self, algorithm_name = "", **kwargs):
        
        self.algorithm = algorithm_name
        self.add_fields(**kwargs)
        self.steps = []   
        
    def add_step(self, **kwargs):
        self.steps.append(kwargs)
        
    def update_step(self, **kwargs):
        for name,value in kwargs.items():
            self.steps[-1][name] = value
        
class ExperimentResult(Result):
    """
    A class to store the results of running an experiment
    """
    
    def __init__(self, experiment_name, **kwargs):
        
        self.experiment = experiment_name
        self.add_fields(**kwargs)
        self.runs = [] 
    
    def add_run(self, result):
        
        self.runs.append(result.__dict__)
