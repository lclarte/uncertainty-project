"""
bo_state_evolution.py
Does state evolution of AMP for the sign perceptron problem
"""

from typing import Tuple
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfc

from core import utility

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)

def H(x : float) -> float:
    '''
    1 - Gaussian cdf.
    '''
    return .5 * erfc(x/np.sqrt(2))

def update_qhat(q : float, int_lims : float = 20.0, sig : float = 0.) -> float:
    '''
    Update of qhat.
    '''
    Delta = sig**2
    V = 1. - q
    def integrand(z):
        # NOTE : In the noiseless case Delta = 0, we recover 2q + V + Delta = 1 + q
        return 1/(np.pi * np.sqrt(V + Delta)) * gaussian(np.sqrt(2*q + V + Delta) * z) * 1/H(np.sqrt(q) * z)
    
    return quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
    
def update_q(x : float) -> float:
    '''
    Update of overlap.
    '''
    return x/(1+x)

def update_se(alpha : float, q: float, sig : float = 0.) -> float:
    '''
    Implements a step the state evolution t -> t+1
    '''
    qhat = alpha * update_qhat(q, sig = sig)
    q_new = update_q(qhat)
        
    return q_new

def iterate_se(alpha : float, max_iter : int = int(1e5), eps : float = 1e-8, init_condition : str ='uninformed', verbose=False, sig = 0., output_list=False, damp=0.5) -> Tuple[float, int]:
    """
    Update state evolution equations. 
    
    Parameters:
    * eps = threshold to reach convergence.
    * max_iter = maximum number of steps if convergence not reached.
    """
    
    # Initialise qu and qv
    q = np.zeros(max_iter)
    
    if init_condition == 'uninformed': 
        q[0] = 0.0001
        
    elif init_condition == 'informed': # -> Bayes optimal since q = 1^-
        q[0] = 0.9999
        
    for t in range(max_iter - 1):
        q_tmp = update_se(alpha, q[t], sig)
        q[t + 1] = utility.damping(q_tmp, q[t], coef_damping=damp)
                
        diff = np.abs(q[t + 1] - q[t])
        
        if verbose:
            print('t: {}, mse: {}'.format(t, 1-q[t + 1]))
        
        if diff < eps:
            break
    
    if t == max_iter - 2:
        print('Escaped BO state evolution because of iteration limit !')
    if output_list:
        return q[:t], t
    else:
        return q[t + 1], t
