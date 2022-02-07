from typing import Tuple
import warnings

from abc import ABC
import numpy as np
from scipy.special import erfc

H_ = lambda x : 0.5 * erfc(x / np.sqrt(2.))

class BayesOptimal(ABC):
    """
    Contains the channel / denoising function used in AMP for the bayes optimal setting 
    where 
    y = sgn(w . x) and w has a gaussian prior
    """
    def gout(w : float, y : int, V : float, sig : float = 0.) -> float:
        """
        Output function (depends on the activation)
        arguments : 
            - sig : STD of noise
        """
        delta = 1e-10
        # Only change this part to take the noise into account
        U = V + sig**2 + delta 
        try:
            deno = np.sqrt(2*np.pi * U) * H_(- y * w / np.sqrt(U)) + delta
            x = y * np.exp(-0.5*(w**2. / U)) / deno
        except Warning:
            print('Error in gout of Bayes')
            return 0
        return x

    def dwgout(w : float, y : int, V : float, sig : float = 0.) -> float:
        """
        Derivative of gout with respect to w
        """
        U = V + sig**2
        delta = 1e-10
        g = BayesOptimal.gout(w, y, V, sig)
        tmp = np.multiply(g, (np.divide(w, U + delta) + g))
        return - np.maximum(tmp, 0.)
        # below : alternative way that looks less stable than the other one
        # return - (w / V) * g - g**2
        
    def channel(y : int, w : float, v : float, sig : float = 0.) -> float:
        return BayesOptimal.gout(w, y, v, sig = sig), BayesOptimal.dwgout(w, y, v, sig = sig)

    def fa(Sigma : float, R : float) -> float:
        """
        Input function, independent of the variance of gaussian prior
        NOTE : Should not depend on the noise in label
        """
        return R / (Sigma + 1.)

    def fv(Sigma : float, R : float) -> float:
        """
        Derivative of input function w.r.t. R, multiplied by Sigma
        """
        return Sigma / (Sigma + 1.)

    def prior(b : float, A : float) -> Tuple[float, float]:
        '''
        Compute f and f' for Bernoulli-Gaussian prior
        
        Sigma = 1 / A
        R = b / A
        '''
        return BayesOptimal.fa(1. / A, b / A), BayesOptimal.fv(1. / A, b / A)
