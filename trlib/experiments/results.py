import json

class Result:
    """
    A class to store the results of running an algorithm
    """
    
    def __init__(self, algorithm_name = "", **kwargs):
        
        self.algorithm = algorithm_name
        for name,value in kwargs.items():
            setattr(self, name, value)
        self.steps = []    
        
    def add_step(self, **kwargs):
        self.steps.append(kwargs)
        
def save_json_callback(file_name):
    """
    Generates a callback for saving results in JSON format
    
    Parameters
    ----------
    file_name: the file where to save results
    
    Returns
    -------
    A callback for an algorithm to save results
    """
    
    def fun(algorithm):
        with open(file_name,"w") as file:
            json.dump(algorithm.result.__dict__,file)
            
    return fun

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