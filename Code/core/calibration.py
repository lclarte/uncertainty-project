# calibration.py
# Functions to compute (empirically and theoretically) the calibration

import numpy as np
from scipy.integrate import nquad, quad
import scipy.linalg
from scipy.special import erfc, erfcinv
import scipy.stats as stats

from core import *
import data
from overlaps import *
import utility

def compute_bo_cond_mean(p : float, qbo : float, qerm : float, Q : float, sigma : float = 0.0) -> float:
    rho = 1.0
    sig_inv_p = utility.sigmoid_inv(p)
    cond_mean= (Q / qerm) * sig_inv_p
    cond_var =  (1.0 - (Q**2 / (qbo * qerm))) * qbo  
    # add sigma squared to account for the noise in the prediction
    return 0.5 * erfc(- cond_mean / np.sqrt(2 * (cond_var + (rho - qbo + sigma**2))))

def compute_bo_cond_variance(p : float, qbo : float, qerm : float, Q : float, sigma : float = 0.0) -> float:
    """
    Return variance of distribution of y b.o conditioned on proba for ERM is p
    """
    rho = 1.0
    sig_inv_p = utility.sigmoid_inv(p)

    cond_mean= (Q / qerm) * sig_inv_p
    cond_var =  (1. - (Q**2 / (qbo * qerm))) * qbo
    std      = np.sqrt(cond_var)

    # compute mean of square : no analytical expression ? 
    def to_integrate(l):
        lprime = cond_mean + (l * std)
        return (0.5 * erfc(- lprime / np.sqrt(2.0 * (rho - qbo + sigma**2))))**2 * stats.norm.pdf(l, loc=0.0, scale=1.0)

    mean_of_square = quad(to_integrate, float('-inf'), float('inf'))[0]

    squared_mean = compute_bo_cond_mean(p, qbo, qerm, Q, sigma)**2    
    return mean_of_square - squared_mean 

def compute_teacher_cond_variance(p : float, rho : float, qerm : float, m : float, sigma : float = 0.0) -> float:
    """
    Return variance of distribution of y b.o conditioned on proba for ERM is p
    """
    rho = 1.0
    sig_inv_p = utility.sigmoid_inv(p)

    cond_mean= (m / qerm) * sig_inv_p
    cond_var =  rho - (m**2 / qerm)
    std      = np.sqrt(cond_var)

    # compute mean of square : no analytical expression ? 
    def to_integrate(l):
        lprime = cond_mean + (l * std)
        return (0.5 * erfc(- lprime / np.sqrt(2.0 * (sigma**2))))**2 * stats.norm.pdf(l, loc=0.0, scale=1.0)

    mean_of_square = quad(to_integrate, float('-inf'), float('inf'))[0]

    squared_mean = (p - compute_teacher_calibration(p, rho, qerm, m, sigma))**2    
    return mean_of_square - squared_mean


def compute_bo_calibration(p : float, qbo : float, qerm : float, Q : float, sigma : float = 0.0) -> float:
    return p - compute_bo_cond_mean(p, qbo, qerm, Q, sigma)

def compute_teacher_bo_calibration(p : float, rho : float, q : float, sigma : float) -> float:
    Delta        = sigma**2
    # NOTE : Expression false if rho not 1 
    bo_inv_p     = - erfcinv(2 * p) * np.sqrt(2*(rho - q + Delta))

    cond_mean_nu = bo_inv_p
    cond_var_nu  = (1. - (q**2 / (rho * q))) * rho

    return p - 0.5 * erfc(- cond_mean_nu / np.sqrt(2. * (cond_var_nu + Delta))) 

def compute_teacher_calibration(p : float, rho : float, qerm : float, m : float, sigma : float) -> float:
    """
    Calibration of ERM with respect to teacher 
    """
    sig_inv_p = utility.sigmoid_inv(p)

    Delta        = sigma**2
    cond_mean_nu = (m / qerm) * sig_inv_p
    cond_var_nu  = (1. - (m**2 / (rho * qerm))) * rho

    return p - 0.5 * erfc(- cond_mean_nu / np.sqrt(2. * (cond_var_nu + Delta))) 

#### EMPIRICAL COMPUTATION OF CALIBRATION FOR EXPERIMENTS

def compute_experimental_teacher_calibration(p, w, werm, Xtest, Ytest, sigma):
    # size of bins where we put the probas
    n, d = Xtest.shape
    dp = 0.025
    Ypred = utility.sigmoid(Xtest @ werm)

    index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
    return p - np.mean([utility.probit(w @ Xtest[i], sigma) for i in index])

def compute_experimental_teacher_variance(p, w, werm, Xtest, Ytest, sigma):
    # size of bins where we put the probas
    n, d = Xtest.shape
    dp = 0.025
    Ypred = utility.sigmoid(Xtest @ werm)

    index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
    return np.var([utility.probit(w @ Xtest[i], sigma) for i in index])

def compute_experimental_bo_calibration(p, what, vhat, werm, Xtest, Ytest, sigma):
    # size of bins where we put the probas
    n, d = Xtest.shape
    dp = 0.025
    Ypred = utility.sigmoid(Xtest @ werm)

    index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
    # add noise of Bayes to the probit
    return p - np.mean([0.5 * utility.erfc(-(what @ Xtest[i]) / np.sqrt(2 * (sigma**2 + 1. - vhat @ Xtest[i]**2))) for i in index])

def compute_experimental_bo_variance(p, what, vhat, werm, Xtest, Ytest, sigma):
    # size of bins where we put the probas
    n, d = Xtest.shape
    dp = 0.025
    Ypred = utility.sigmoid(Xtest @ werm)

    index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
    # add noise of Bayes to the probit
    return np.var([0.5 * utility.erfc(-(what @ Xtest[i]) / np.sqrt(2 * (sigma**2 + 1. - vhat @ Xtest[i]**2))) for i in index])
