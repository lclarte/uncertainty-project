# joint_density.py
# Used to compute the joint probability distribution between ERM and BO

import itertools
import sys
from typing import Tuple

from tqdm.utils import disp_len

sys.path.append('core')
sys.path.append('experiments')

import matplotlib as mpl
import matplotlib.pyplot as plt
import json

import numpy as np
import pandas as pd
from scipy.integrate import nquad, quad
import scipy.linalg
from scipy.special import erfc, erfcinv
import scipy.stats as stats
from tqdm import tqdm

import constants, datalogging
from core.utility import * 
from core.overlaps import Overlaps

from sacred.observers import FileStorageObserver
from sacred import Experiment

ex = Experiment('joint_density_experiment')

def load_data(load : str) -> Tuple[float, float, float, Overlaps]: 
    """
    Load the overlaps from the specified folder
    """
    configuration, results = datalogging.load_config_and_result(load)
        
    alpha = configuration['alpha']
    lamb  = configuration['lambda']
    sigma   = configuration['sigma']

    content = results['overlaps']

    overlaps = Overlaps(
        rho  = content['rho'],
        V    = content['V'],
        Q    = content['Q'],
        m    = content['m'],
        mbo  =  content['mbo'],
        qbo  =  content['qbo'],
        qerm = content['qerm']
    )

    return alpha, lamb, sigma, overlaps

# ======= 

def integrand_p_correct(nu, cov, qbo, a, b, sigma = 0.0):
    y = np.sign(nu)
    Delta = sigma**2
    alpha = - y / np.sqrt(2 * (1. - qbo + Delta))

    inv_l =  erfcinv(2 * a) / alpha
    inv_lerm = y * sigmoid_inv(b)

    normalisation_bo = (alpha / 2.) * erfc_prime(alpha * inv_l)
    normalisation_erm = b * (1. - b)

    return stats.multivariate_normal.pdf([nu, inv_l, inv_lerm], mean=np.zeros(3), cov=cov, allow_singular=False) / np.abs(normalisation_bo * normalisation_erm)

def get_p_correct(a, b, qbo, Sigma, sigma = 0.0):
    integral = quad(lambda nu : integrand_p_correct(nu, Sigma, qbo, a, b, sigma), -float('inf'), float('inf'), limit=100)
    return integral[0]

def get_p_correct_density(cov, qbo, N, sigma = 0.0, vmin=0., vmax=1.):
    dx = 1. / N
    subdivisions = np.linspace(vmin + dx / 2., vmax - dx / 2., N)
    
    matrix = np.zeros((N, N))
    with tqdm(total=N * N) as pbar:
        # i is for B.O, j is for ERM
        for i in range(N):
            for j in range(N):
                # TODO : Implementer la densité pour calculer P = 1 => calibration
                a, b = subdivisions[i], subdivisions[j]
                # divide by area s.t the integral over all square is 1 
                matrix[i, j] = get_p_correct(a, b, qbo, cov, sigma)
                pbar.update(1)
    return matrix

# ========= 

def get_p_one(a, b, qbo, Sigma_2, sigma : float = 0.):
    """
    Between ERM and Bayes
    """
    Delta = sigma**2
    alpha = - 1 / np.sqrt(2 * (1. - qbo + Delta))

    inv_l =  erfcinv(2 * a) / alpha
    inv_lerm = sigmoid_inv(b)

    normalisation_bo = (alpha / 2.) * erfc_prime(alpha * inv_l)
    normalisation_erm = b * (1. - b)

    return stats.multivariate_normal.pdf([inv_l, inv_lerm], mean=np.zeros(2), cov=Sigma_2, allow_singular=False) / np.abs(normalisation_bo * normalisation_erm)

def get_p_one_density(Sigma_2, qbo, N, sigma = 0.0, vmin=0., vmax=1.):
    # row is for bayes, column is for ERM
    dx = 1. / N
    subdivisions = np.linspace(vmin + dx / 2., vmax - dx / 2., N)
    
    matrix = np.zeros((N, N))
    with tqdm(total=N * N) as pbar:
        # i is for B.O, j is for ERM
        for i in range(N):
            for j in range(N):
                # TODO : Implementer la densité pour calculer P = 1 => calibration
                a, b = subdivisions[i], subdivisions[j]
                # divide by area s.t the integral over all square is 1             
                matrix[i, j] = get_p_one(a, b, qbo, Sigma_2, sigma)
                pbar.update(1)
    return matrix

# === Between ERM and teacher 

def get_p_one_teacher(a, b, rho, Sigma_2, sigma : float = 0.):
    """
    Between ERM and teacher
    """
    assert sigma > 0
    Delta = sigma**2
    # If Delta = 0.0, no density is defined for the teacher
    alpha = - 1 / np.sqrt(2 * Delta)

    inv_l =  erfcinv(2 * a) / alpha
    inv_lerm = sigmoid_inv(b)

    normalisation_teacher = (alpha / 2.) * erfc_prime(alpha * inv_l)
    normalisation_erm = b * (1. - b)

    return stats.multivariate_normal.pdf([inv_l, inv_lerm], mean=np.zeros(2), cov=Sigma_2, allow_singular=False) / np.abs(normalisation_teacher * normalisation_erm)

def get_p_one_teacher_density(Sigma_2, rho, N, sigma = 0.0, vmin=0., vmax=1.):
    # row is for teacher, column is for ERM
    dx = 1. / N
    subdivisions = np.linspace(vmin + dx / 2., vmax - dx / 2., N)
    
    matrix = np.zeros((N, N))
    with tqdm(total=N * N) as pbar:
        # i is for B.O, j is for ERM
        for i in range(N):
            for j in range(N):
                # TODO : Implementer la densité pour calculer P = 1 => calibration
                a, b = subdivisions[i], subdivisions[j]
                # divide by area s.t the integral over all square is 1             
                matrix[i, j] = get_p_one_teacher(a, b, rho, Sigma_2, sigma)
                pbar.update(1)
    return matrix

# ==== Between Bayes and the teacher 

def get_p_one_bo_teacher(a, b, sigma, Sigma_bo_teacher):
    """
    NOTE : Here, we need to add the noise to bayes otherwise it makes no sense
    NOTE 2 : Use teacher WITHOUT noise in the covariance matrix (it 's surely 
    equivalent the computations are probably more complicate in that case)
    First arg is for teacher, second is for bayes
    """
    if Sigma_bo_teacher.shape != (2, 2):
        raise Exception()

    qbo = Sigma_bo_teacher[1, 1]

    Delta    = sigma**2
    alpha_bo = - 1 / np.sqrt(2 * (1. - qbo + Delta))
    alpha_teacher = - 1 / np.sqrt(2 * Delta)

    inv_teacher =  erfcinv(2 * b) / alpha_teacher
    inv_bo      =  erfcinv(2 * a) / alpha_bo

    normalisation_bo = (alpha_bo / 2.) * erfc_prime(alpha_bo * inv_bo)
    normalisation_teacher = (alpha_teacher / 2.) * erfc_prime(alpha_teacher * inv_teacher)

    return stats.multivariate_normal.pdf([inv_teacher, inv_bo], mean=np.zeros(2), cov=Sigma_bo_teacher, allow_singular=False) / np.abs(normalisation_bo * normalisation_teacher)

def get_p_one_bo_teacher_density(sigma, Sigma_bo_teacher, N, vmin=0., vmax=1., bo_on_rows = True):
    """
    NOTE : rows will be for bayes, columns for teacher 
    """
    dx = 1. / N
    subdivisions = np.linspace(vmin + dx / 2., vmax - dx / 2., N)
    
    matrix = np.zeros((N, N))
    with tqdm(total=N * N) as pbar:
        # i is for B.O, j is for ERM
        for i in range(N):
            for j in range(N):
                # TODO : Implementer la densité pour calculer P = 1 => calibration
                a, b = subdivisions[i], subdivisions[j]
                # divide by area s.t the integral over all square is 1             
                matrix[i, j] = get_p_one_bo_teacher(a, b, sigma, Sigma_bo_teacher)
                pbar.update(1)
    if bo_on_rows == False:
        return matrix.T
    return matrix

@ex.config
def config():
    # dimension not used for the theoretical prediciton but for experiments
    d                = 1000

    # can be be 'p_correct' (for the proba to be correct), 'p_one' (proba to be equal to 1) or 'variance' 
    quantity         = 'p_correct'
    method           = 'mc'

    load_file        = ''
    # by default, we give the same file 
    save_folder      = constants.DEFAULT_JOINT_DENSITY_FOLDER
    save_file        = constants.DEFAULT_JOINT_DENSITY_FILE_NAME
    verbose          = False

    # define boundaries for computation. Useful when dist. is peaked
    # near (1, 1)
    vmin             = 0.
    vmax             = 1.
    N                = 20

    dic              = dict(locals())

@ex.automain
def main(dic):
    N    = dic['N']
    vmin = dic['vmin']
    vmax = dic['vmax']
    quantity = dic['quantity']

    da = db = 1. / N
    subdivisions = np.linspace(vmin + da / 2., vmax - db / 2., N)
    
    alpha, lamb, sigma, overlaps = load_data(dic['load_file'])

    Sigma = overlaps.get_teacher_bo_erm_covariance(sigma, add_noise=False)
    det = np.linalg.det(Sigma)
    if det <= 0:
        print('det of Sigma : ', det)
        raise Exception('Matrix must be positive definite')

    if quantity == 'p_correct':
        matrix = get_p_correct_density(Sigma, overlaps.qbo, N, vmin, vmax)

    if dic['verbose'] == True:
        print('Sum of elements : ', np.sum(matrix))

    config = {
        'alpha' : alpha,
        'lamb'  : lamb,
        'sigma'  : sigma,
        'N'    : N,
        'vmin' : vmin,
        'vmax' : vmax,
        'quantity' : quantity,
    }

    result = {
        'density' : matrix.tolist()
    }

    save_folder = dic['save_folder']
    if save_folder[-1] != '/':
        save_folder += '/'
    save_file = dic['save_file']

    save_file = save_folder + datalogging.get_time_default_name(save_file)
    datalogging.save_config_and_result(config, result, save_file)