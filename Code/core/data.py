'''
data.py
File containing the functions to generate the data
So far, only data w/ gaussian covariance matrix
NOTE : For now, all 
'''

from typing import Tuple
import numpy as np
import sys

from numpy.core.fromnumeric import size
sys.path.append('..')
sys.path.append('core')

import core.utility as utility

def iid_teacher(d : float) -> np.ndarray:
    return np.random.normal(0., 1., size=(d, ))

def iid_input(n : int, d : float) -> np.ndarray:
    return np.random.normal(0., 1., size=(n, d)) / np.sqrt(d)

class DataSampler:
    def __init__(self, **kwargs) -> None:
        # by default, we are deterministic (also to be backward compatible)
        if 'sig' in kwargs:
            self.sig = kwargs['sig']
        else:
            self.sig = 0.

    def sample_instance(self, d : int, alpha : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0 = iid_teacher(d)
        F, y = self.sample_data(x0, alpha)
        return x0, F, y

    def sample_data_n(self, w0 : np.ndarray, n : int) -> Tuple[np.ndarray, np.ndarray]:
        d = len(w0)
        F = iid_input(n, d)
        return F, self.activation(F @ w0)

    def sample_data(self, w0 : np.ndarray, alpha : float) -> Tuple[np.ndarray, np.ndarray]:
        d = len(w0)
        n = int(np.ceil(alpha * d))
        return self.sample_data_n(w0, n)
    
    def activation(self, preactivation : np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LogitModelData(DataSampler):
    def activation(self, preactivation: np.ndarray) -> np.ndarray:
        assert self.sig >= 0
        if self.sig == 0.: 
            return np.sign(preactivation)
        n = len(preactivation)
        probas = utility.sigmoid(preactivation / self.sig)
        return 2*np.ceil(probas - np.random.uniform(size=n)) - 1.

class ProbitModelData(DataSampler):
    def activation(self, preactivation : np.ndarray)-> np.ndarray:
        n = len(preactivation)
        return np.sign(preactivation + self.sig * np.random.normal(0., 1., size=(n, )))

def predict_label(X : np.ndarray, w : np.ndarray) -> np.ndarray:
    '''
    For classification tasks only 
    DONT PUT THE NORMALIZATION HERE, IT'S DONE AT SAMPLING
    '''
    return np.sign(X@w)

# NOTE : We keep these functions not to break the rest of the (old) code 

# dictionnaire pour pouvoir faire la conversion
data_classes_names = {
    'logit'   : LogitModelData,
    'probit'  : ProbitModelData,
}

def sample_instance(d : int, alpha : float, data_model : str, **kwargs):
    """
    Sample instance of the problem.
    Samples F from P(F) and {x, y} from P(x, y | F)
    """
    sampler = data_classes_names[data_model](**kwargs)
    return sampler.sample_instance(d, alpha)

def sample_data(w0 : np.ndarray, alpha : float, data_model : str, **kwargs):
    sampler = data_classes_names[data_model](**kwargs)
    return sampler.sample_data(w0, alpha)

def sample_data_n(w0 : np.ndarray, n : int, data_model : str, **kwargs):
    sampler = data_classes_names[data_model](**kwargs)
    return sampler.sample_data_n(w0, n)