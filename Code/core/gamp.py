# gamp.py
# File containing the code for GAMP with sign perceptron and gaussian prior

from typing import List
import numpy as np
from scipy.special import erfc
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import pandas as pd

import data

from models.bayes_optimal import BayesOptimal
from models.amp_erm import ERM

# Functions specific to the prior and activation function 

def check_increasing(mses : List[float], epochs : int =5) -> bool:
    if epochs > len(mses):
        print('Number of epochs must be smaller than length of array!')
        return False
    else:
        return True if np.all(np.diff(mses[-epochs:]) > 0) else False

# G-AMP

def iterate_gamp(W : List[List[float]], y : List[float], x0 : List[float] =None, model : type = BayesOptimal, max_iter : int =200, tol : float =1e-7, 
                 damp : float =0.2, early_stopping : bool =False, verbose : bool = True, sig : float = 0.) -> dict:
    """
    MAIN FUNCTION : Runs G-AMP and returns the finals parameters. If we study
    the variance, we are interested in the vhat quantities. The 'variance' of the vector 
    w will (normally) be the sum of the vhat.

    parameters :
        - W : data matrix
        - y : funciton output
        - x0 : ground truth
    returns : 
        - retour : dictionnary with informations
    """
    assert not x0 is None
    d = len(x0)

    # Preprocessing
    y_size, x_size = W.shape
    W2 = W * W
    
    # Initialisation
    xhat = np.zeros(x_size)
    vhat = np.ones(x_size)
    g = np.zeros(y_size)

    count = 0
    mses = np.zeros(max_iter)

    status = None

    q_list, m_list = [], []

    for t in range(max_iter):

        q = np.mean(xhat**2)
        m = np.mean(xhat * x0)
        q_list.append(q)
        m_list.append(m)

        if verbose:
            print(f'q = {q} and m = {m}. Relative difference is {np.abs(m - q) / q}')

        # First part: m-dimensional variables
        V     = W2 @ vhat
        # here we see that V is the Onsager term
        omega = W @ xhat - V * g
        g, dg = model.channel(y, omega, V, sig = sig)
        
        # Second part
        A = -W2.T @ dg
        b = A*xhat + W.T @ g
        
        xhat_old = xhat.copy() # Keep a copy of xhat to compute diff
        vhat_old = vhat.copy()

        xhat, vhat = model.prior(b, A)

        diff = np.mean(np.abs(xhat-xhat_old))
        # Expression of MSE has been changed
        
        mses[t] = 1. - np.mean(xhat * x0)
            
        if count == 5:
            status = 'Early stopping'
            return mses[:t-4]
        
        # if verbose:
        #    print('t: {}, diff: {}, mse: {}'.format(t, diff, mses[t]))
        
        if (diff < tol) or (mses[t] < tol):
            status = 'Done'
            break

    if verbose:
        print('t : ', t)

    retour = {}
    retour['mse'] = mses[t]
    retour['status'] = status
    retour['estimator'] = xhat
    retour['variances'] = vhat
    retour['q_list'] = q_list
    retour['m_list'] = m_list
    
    return retour