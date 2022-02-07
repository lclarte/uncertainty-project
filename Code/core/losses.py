'''
losses.py
Includes the losses / error functions for different models and tasks
'''

import sys

from data import predict_label
import numpy as np
from scipy.integrate import nquad
from scipy.stats     import multivariate_normal

# === LOSSES FOR AMP WITH THE ORDER PARAMETERS

def mse(x : np.ndarray, xhat : np.ndarray) -> float:
    '''
    Compares the estimator and the ground truth parameter
    '''
    return np.mean((x - xhat)**2)

def logistic_loss(y : np.ndarray, yhat : np.ndarray) -> float:
    """
    Remark : yhat can be preactivation
    """
    loss = lambda z : np.log(1 + np.exp(-z))
    return np.mean(loss(yhat * y))

def square_loss(y : np.ndarray, yhat : np.ndarray):
    return np.mean((y - yhat)**2)

def classification_error(y : np.ndarray, yhat : np.ndarray) -> float:
    '''
    Error for the classfication task, hence the proportion factor
    yhat and y must be -1 or 1
    '''
    # equivalent to y != yhat since they take only 2 values -1 or 1
    return 0.25 * square_loss(y, yhat)

def classification_error_overlap(q : float, rho : float =1., sig :float = 0.) -> float:
    """
    NOTE : On peut simplement utiliser la formule avec l'arccos, non ?
    NOTE : Usable only in the bayes optimal setting !! 
    arguments : 
        - q : teacher-student overlap
        - rho : teacher-teacher overlap
    """
    Delta = sig**2
    covariance = np.array([[rho + Delta, q],[q, q]])
    mean = np.array([0., 0.])

    ranges = [(float('-inf'), float('inf')), (float('-inf'), float('inf'))]

    res = nquad(lambda x, y : float(np.sign(x) != np.sign(y)) * multivariate_normal.pdf([x, y], mean=mean, cov=covariance), ranges)
    return res[0]

'''
Below : dictionnaries relating the task to their corresponding loss functions
NOTE : For ridge, we could express the generalizsation error analytically as
a function of w and what, but for now we compute the error by sampling a test set
'''

# Training losses 
dic_losses = {
    'logistic' : logistic_loss,
    'ridge'    : square_loss,
    'l2_classification' : square_loss,
}

# Remember, we'll alway use the preactivations as input ! 
# They must be already normalized by d 
dic_error = {
    'ridge'    : square_loss,
    'logistic' : lambda y, yhat : classification_error(np.sign(y), np.sign(yhat)),
    'l2_classification' : lambda y, yhat : classification_error(predict_label(y), predict_label(yhat))
}
