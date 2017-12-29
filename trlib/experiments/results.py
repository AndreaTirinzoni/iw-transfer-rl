import json

class Result:
    """
    A class to store the results of running an algorithm
    """
    
    def __init__(self, algorithm_name = "", **kwargs):
        
        self.algorithm = algorithm_name
        self.add_fields(**kwargs)
        self.steps = []   
        
    def add_fields(self, **kwargs):
        for name,value in kwargs.items():
            setattr(self, name, value) 
        
    def add_step(self, **kwargs):
        self.steps.append(kwargs)
        
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