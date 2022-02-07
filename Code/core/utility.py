# utility.py
# various useful functions

from typing import List
import numpy as np
import scipy.optimize as opt4
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import erf, erfc
from datetime import datetime

# FUNCTION OF THE LOGIT MODEL 
sigmoid = np.vectorize(lambda x : 1. / (1. + np.exp( -x )))
sigmoid_inv = np.vectorize(lambda y : np.log(y/(1-y)))

erf_prime = np.vectorize(lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2))
erfc_prime = np.vectorize(lambda x : -2. / np.sqrt(np.pi) * np.exp(-x**2))

bernoulli_variance = np.vectorize(lambda p : 4 * p * (1. - p))

def probit(lf, sigma):
    return 0.5 * erfc(- lf / np.sqrt(2 * sigma**2))

# Different kinds of variances

def compute_vector_variance(w_list : List[np.ndarray]) -> float:
    '''
    NOTE : Outdated
    Compute the quantity E( ||w||^2 ) - || E(w) ||^2 
    '''
    d = len(w_list[0])
    w_array = np.array(w_list)
    vars = np.var(w_array, axis=0)
    return np.mean(vars)

def compute_variances(w_list : List[np.ndarray]) -> dict:
    '''
    NOTE : Outdated
    Argument : 
        - w_list : N x d matrix 
    Returns : 
        - variance of w as computed by 'compute_vector_variance'
        - variance of the squared norm of w, renormalized by d
    '''
    d = len(w_list[0])
    w_list = np.array(w_list)
    sqd_norm = np.sum(w_list**2, axis=1)
    return {
        'w_variance' : compute_vector_variance(w_list),
        'sqd_norm_variance' : np.var(sqd_norm) / d
    }

def proximal_operator_by_derivation(func : callable, func_prime : callable, x : float, tau : float) -> float:
    """
    Returns the root of func_prime + (z - x) / tau
    Need func_prime to use newton
    """
    to_root = lambda z : (z - x )/ tau + func(z)
    to_root_prime = lambda z : 1 / tau + func_prime(z)
    res = root_scalar(to_root, x0=x, fprime=to_root_prime)
    return res.root

def proximal_operator(func : callable, x : float, tau : float) -> float:
    to_minimize = lambda z : ((z - x)**2) / (2 * tau) + func(z)
    res = minimize_scalar(to_minimize, method='Golden')
    if res['x'] > 1e10:
        print(res['x'])
    return res['x']
    # res.x here is an array with a single element inside

# ==== VARIANCE FOR THE BAYES-OPTIMAL VARIANCE ==== 

def Zy(y : int, w : float, V : float, sigma : float = 0.0) -> float:
    delta = 1e-10
    U = V + sigma**2 + delta 
    return 0.5 * (1 + erf(y * w / np.sqrt(2*U)))

def y_teacher_proba(w : List[float], x : List[float], y : int, sigma : float = 0.) -> float:
    """
    Only works for y = -1 or 1
    """
    return Zy(y, w @ x, 0.0, sigma)


def y_bo_proba(what : List[float], vhat : List[float], x : List[float], y : int, sigma : float = 0.) -> float:
    """
    Only works for y = -1 or 1
    """
    return Zy(y, what @ x, vhat @ x**2, sigma)

def y_bo_expectation(what : List[float], vhat : List[float], x : List[float], sigma : float = 0.) -> float:
    return y_bo_proba(what, vhat, x, 1, sigma) - y_bo_proba(what, vhat, x, -1, sigma)

def y_bo_variance(what : np.ndarray, vhat : np.ndarray, x : List[float], sigma : float = 0.0) -> float:
    # reminder : expectation of y^2 is 1 because of binary labels
    expectation = y_bo_expectation(what, vhat, x, sigma)
    return 1. - expectation**2

# ==== INTEGRATE WITH MONTE CARLO W.R.T GAUSSIAN MEASURE ==== 

def gaussian_mc(func : callable, mean : List[float], cov : List[List[float]], n_samples : int = 10000) -> float:
    tmp = []
    Xs = np.random.multivariate_normal(mean, cov, size=(n_samples,))
    for i in range(n_samples):
        tmp.append(func(Xs[i]))
    return np.mean(tmp)

# === DAMPING USED E.G. to compute 

def damping(q_new : float, q_old : float, coef_damping : float =0.5) -> float:
    if q_old == float('inf') or np.isnan(q_old):
        return q_new
    return (1 - coef_damping) * q_new + coef_damping * q_old
