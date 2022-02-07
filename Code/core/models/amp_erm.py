import sys
from time import time

from abc import ABC
from typing import Tuple
import scipy.optimize as opt
import numpy as np

sys.path.append('..')
sys.path.append('core')
import utility

class ERM:
    """
    NOTE : This code should not be changed in the noiseless case as it's not aware of the noise
    and explicitely depends on the input data
    """
    
    proximal_by_derivation : bool = False

    def __init__(self, lamb : float = 1.) -> None:
        self.lamb = lamb

    def gout(self, w : float, y : int, V : float, **kwargs) -> float:
        logistic = lambda x : np.log(1. + np.exp(-y*x))
        logistic_prime = lambda x : - y / (1. + np.exp(y * x))
        logistic_second = lambda x : - (y**2) * np.exp(y  * x) / (1. + np.exp(y * x))**2
        # should be correct
        try:
            if self.proximal_by_derivation:
                prox = utility.proximal_operator_by_derivation(logistic_prime, logistic_second, w, V)
            else:
                prox = utility.proximal_operator(logistic, w, V)
        except RuntimeWarning as e:
            print(e)
            print('w, y, V = ', w, y, V)
            return 0.
        return (1. / V) * (prox - w)
        
    def dwgout(self, w : float, y : int, V : float , f : float = None, **kwargs) -> float:
        # do not recompute the proximal operator twice, reuse previous computations
        f = f or self.gout(w, y, V)
        # On peut enlever le y du cosh par symmetrie de la fonction
        alpha = (2. * np.cosh(0.5 * y * (w + V*f)))**2
        # Sanity check : apparement, pour que le onsager tBaseERM soit globalement positif, il faut que dwgout soit negatif
        return - 1. / (alpha + V )

    def channel(self, y : int, w : float, v : float, **kwargs) -> Tuple[float, float]:
        """
        NOTE : We completely discard the sig. (noise) but keep it so the signature is the same as AMP
        and avoid bugs
        """
        # Need to do it for each coordinate 
        n = len(w)
        g, dg = np.zeros_like(y), np.zeros_like(y)
        for i in range(n):
            g[i] = self.gout(w[i], y[i], v[i])
            dg[i] = self.dwgout(w[i], y[i], v[i], f = g[i])
        return g, dg

    # for the L2 regularization, the functions are the same as the gaussian prior

    def fa(self, Sigma : float, R : float) -> float:
        """
        Input function
        """
        return R / (self.lamb * Sigma + 1.)

    def fv(self, Sigma : float, R : float) -> float:
        """
        Derivative of input function w.r.t. R, multiplied by Sigma
        """
        return Sigma / (self.lamb * Sigma + 1.)

    def prior(self, b : float, A : float) -> Tuple[float, float]:
        """
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        """
        return self.fa(1. / A, b / A), self.fv(1. / A, b / A)

