'''
erm.py
Compute exact ERM on logistic or ridge regression w/ sklearn 
'''

from typing import List, Dict 

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from data import *
import losses

def get_errors(w0 : List[float], X : List[List[float]], y : List[float], w : List[float], task : str, data_model : str, sig : float = 0.) -> Dict[str, float]:
    n = len(X)
    alpha = float(n) / len(w0)
    # sample as many data than training data
    X_test, y_test = sample_data(w0, n, data_model, sig = sig)
    # Always use the preactivations ! 
    yhat_test, yhat_train = X_test @ w, X @ w

    train_error = losses.dic_error[task](y, yhat_train)
    # Note that we could compute the generalisation error with a closed form expression
    test_error  = losses.dic_error[task](y_test, yhat_test)
    train_loss  = losses.dic_losses[task](y, yhat_train)

    mse = losses.mse(w0, w)

    return {'train_error' : train_error,
             'test_error' : test_error,
             'train_loss' : train_loss,
             'mse'        : mse}

# Functions to do the regressions

def erm_ridge_regression(X : List[List[float]], y : List[float], lamb : float =1.) -> List[float]:
    lr = Ridge(alpha = lamb, fit_intercept=False, tol=1e-7)
    lr.fit(X, y)
    return lr.coef_[0]

def erm_logistic_regression(X : List[List[float]], y : List[float], lamb : float =1.) -> List[float]:
    """
    Logistic regrsssion with L2 penalty"""
    # CAUTION : Cannot do train / test split because that would change the ratio alpha
    # so we will sample the training set independently
    lamb = float(lamb)
    max_iter = 10000
    tol      = 1e-16

    if lamb > 0.:
        lr = LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, 
                                  C = (1. / lamb), max_iter=max_iter, tol=tol, verbose=0)
    else:
        lr = LogisticRegression(penalty='none',solver='lbfgs',fit_intercept=False, max_iter=max_iter, tol=tol, verbose=0)
    lr.fit(X, y)

    if lr.n_iter_ == max_iter:
        print('Attention : logistic regression reached max number of iterations ({:.2f})'.format(max_iter))

    w = lr.coef_[0]
    return w
