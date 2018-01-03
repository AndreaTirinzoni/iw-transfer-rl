import pickle

def save_object(obj, file_name):
    """
    Saves an object to a file in .pkl format
    
    Parameters
    ----------
    file_name: the file where to save (without extension)
    """
    with open(file_name + ".pkl", 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
    """
    Loads an object from a file in .pkl format
    
    Parameters
    ----------
    file_name: the file where to load (without extension)
    
    Returns
    -------
    The loaded object
    """
    
    with open(file_name + ".pkl", 'rb') as file:
        return pickle.load(file)