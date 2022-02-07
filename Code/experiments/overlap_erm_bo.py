from enum import Enum, auto
import json, sys
from tabnanny import verbose
from time import time
from typing import List, Tuple

from mpi4py import MPI
import numpy as np
from numpy import linalg, sqrt, square
from sacred import Experiment
from scipy import stats
from scipy.integrate import nquad, quad
import scipy.linalg
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


sys.path.append('core')
sys.path.append('experiments')
sys.path.append('../Libraries')
improved_se = __import__("state-evolution-erm-logistic")

from core.models.bayes_optimal import BayesOptimal
from core.models.amp_erm import ERM

from core import bo_state_evolution, gcm_erm, utility, data, gamp, erm
from core.overlaps import Overlaps, average_overlaps
import constants, datalogging

# =========== END OF IMPORTS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ex = Experiment('overlap_erm_bo_experiment')

# NOTE : Temporairement, on supprime le code pour la prediction théorique des overlaps
# c'est plus simple

class ExactIterationMethod(Enum):
    NQUAD = auto()
    MC    = auto()

def get_theoretical_separate_overlap(alpha : float, lamb : float, sig : float, d : int) -> Overlaps:
    theta = np.random.normal(0., 1., d)

    res_erm = gcm_erm.compute_se(d, alpha, sig, lamb, theta) 
    V, qerm, m = res_erm['overlaps']['variance'], res_erm['overlaps']['self_overlap'], res_erm['overlaps']['teacher_student']
    hatV, hatqerm, hatm = res_erm['hatV'], res_erm['hatq'], res_erm['hatm']
    qbo, _ = bo_state_evolution.iterate_se(alpha, sig = sig, eps=1e-8)
    hatqbo = bo_state_evolution.update_qhat(qbo, sig = sig)
    rho = res_erm['rho']

    return Overlaps(
        rho  = rho,
        qbo  = qbo,
        qerm = qerm,
        m    = m,
        V    = V,
        hatqbo = hatqbo,
        hatqerm = hatqerm,
        hatV = hatV,
        hatm = hatm,
    )

def get_theoretical_separate_overlap_reduced(alpha : float, lamb : float, sig : float, precision : float = 1e-5, code_version : str = 'ben', d = 1000) -> Overlaps:
    # NOTE : Only get the bare min. of overlaps to compute the calibration
    if code_version == 'ben':
        res = improved_se.main.solve_SE(alpha = alpha, lambd = lamb, sigma = sig, verbose = False, precision = 1e-7)
        m = res['m']
        qerm = res['q']
    elif code_version == 'gcm':
        theta = data.iid_teacher(d)
        res = gcm_erm.compute_se(d, alpha, sig, lamb, theta)
        m = res['overlaps']['teacher_student']
        qerm = res['overlaps']['self_overlap']
    else:
        raise Exception()

    qbo, _ = bo_state_evolution.iterate_se(alpha, sig = sig, eps=1e-10)
    rho = 1.0

    return Overlaps(
        rho  = rho,
        qbo  = qbo,
        qerm = qerm,
        m    = m,
    )

def get_average_theoretical_separate_overlap_reduced(alpha : float, lamb : float, sig : float, ntrials : int, precision : float = 1e-5, code_version : str = 'ben', d = 1000) -> Overlaps:
    ov_list = []
    for i in range(ntrials):
        ov_list.append(get_theoretical_separate_overlap_reduced(alpha, lamb, sig, precision=precision, code_version=code_version, d=d))
    return average_overlaps(ov_list)

def get_experimental_overlap(alpha : float, lamb : float, sig : float, d : int, data_model) -> Overlaps:
    """
    As a function of alpha, run BO and ERM and compute the overlap between the 2
    """
    w, X, y = data.sample_instance(d, alpha, data_model, sig = sig)
    # run AMP
    res_gamp = gamp.iterate_gamp(X, y, w, verbose = 0, sig = sig)
    wbo      = res_gamp['estimator']
    # run erm
    w_erm = erm.erm_logistic_regression(X, y, lamb)
    ovrlp = np.mean(w_erm * wbo)

    overlaps = Overlaps(
        qbo = np.mean(wbo * wbo),
        qerm= np.mean(w_erm * w_erm),
        m   = np.mean(w_erm * w),
        # Used because of finite size effects
        mbo = np.mean(w * wbo),
        rho = np.mean(w * w),
        # cannot be computed easily empirically
    )

    overlaps.Q = ovrlp
    return overlaps

def get_overlaps(alpha : float, lamb : float, sig : float, experimental : bool,
                 d : int =1000, data_model : str ='probit', th_method : ExactIterationMethod =ExactIterationMethod.MC, 
                 mc_samples : int = 10**3, tol : float =1e-2, verbose : bool =False) -> Overlaps:
    """
    Does ONE trial (we need to do several)
    NOTE : dic is only useful here for the theoretical overlaps, so can be void if 
    we want the experimental overlaps.
    In that case, dic must contain : 'sig', 'mc_samples', 'methdod'
    """
    # get_experimental_overlap to actually run AMP and get the results
    if experimental:
        overlaps = get_experimental_overlap(alpha, lamb, sig, d, data_model)
    else:
        raise NotImplementedError

    return overlaps

def get_average_overlaps(alpha : float, lamb : float, sig : float, experimental : bool, 
                         ntrials : int , d : int =1000, data_model : str ='probit', th_method : ExactIterationMethod = ExactIterationMethod.MC,
                         mc_samples : int = 10**3, tol : float = 1e-2, verbose : bool =False) -> Overlaps:
    # don't need the hat overlaps here
    overlaps : List[Overlaps] = []

    # do 1 run with experimental to check the overlaps are finite
    finite_overlaps_check = get_overlaps(alpha, lamb, sig, True, d, data_model, th_method, mc_samples, tol, verbose=False)
    threshold = 1000
    if finite_overlaps_check.qerm > threshold:
        # theory won't work in this case
        experimental = True

    for i in range(ntrials):
        if verbose:
            print('Trial : ', i+1)
        result = get_overlaps(alpha, lamb, sig, experimental, d, data_model, th_method, mc_samples, tol, verbose)
        overlaps.append(result)
    return average_overlaps(overlaps)

def generate_alpha_lambda_pairs(alpha, lambda_error, lambda_r2_score): 
    return [(a, l, 'min-error') for a, l in zip(alpha, lambda_error)] + [(a, l, 'max-score') for a, l in zip(alpha, lambda_r2_score)]

# LOAD AND SAVE THE RESULTS

def load_results(load_file, dic): 
    """
    Contains the result of the script lambda_search.py
    arguments : 
        - dic : the dictionnary will be updated with the new information
    """
    copy_dict = dict(dic)
    
    configurations, results = datalogging.load_config_and_result(load_file)

    copy_dict['alpha'] = configurations['alpha_range']
    copy_dict.update(results)
    return copy_dict

# TODO : Fonction qui essaie de charger les overlaps, les calcule si elle ne peut pas 
# et le cas echeant les sauvegarde dans le bon fichier 

@ex.config
def config():
    # NOTE : Useless for now

    # RELATIVE error tolerance
    tol          = 1e-2
    experimental = True
    # 'nquad', 'mc'
    method       = 'mc'
    mc_samples   = 50000
    
    ntrials      = 1
    verbose      = False

    # NOTE : Below are the parameters that can be loaded from lambnda_search experiment
    data_model = 'probit'    
    sigma      = 0.
    d          = 1000
    # sample complexity : either one or a full range 
    alpha : List[float] = [ 0.1 ]
    lamb  : List[float] = [ 0. ]

    load_file = ''
    # specify folder cuz we'll have several files to save ... 
    save_folder = constants.DEFAULT_OVERLAPS_FOLDER
    save_file   = constants.DEFAULT_OVERLAPS_FILE_NAME
    
    dic = dict(locals())

@ex.automain
def main(dic):
    # so dic is not read only 
    dic = dict(dic)
    if dic['load_file'] != '':
        dic = load_results(dic['load_file'], dic)
        alpha_lambda_pairs = generate_alpha_lambda_pairs(dic['alpha'], dic['lambda_error'], dic['lambda_r2_score'])
    
    else:
        assert len(dic['alpha']) == len(dic['lamb']), "Length of alpha / lamb are {} and {}".format(len(dic['alpha']), len(dic['lamb']))
        alpha_lambda_pairs = [(a, l, '{:.5f}'.format(l)) for (a, l) in zip(dic['alpha'], dic['lamb'])]
    
    del dic['alpha'], dic['lamb']

    npairs = len(alpha_lambda_pairs)

    
    ntrials      = dic['ntrials']
    d            = dic['d'] 
    data_model   = dic['data_model']
    sigma        = dic['sigma']
    experimental = dic['experimental']
    if dic['method'] == 'mc':
        th_method    = ExactIterationMethod.MC
    elif dic['method'] == 'nquad':
        th_method    = ExactIterationMethod.NQUAD
    else:
        raise Exception()
    mc_samples   = dic['mc_samples']
    tol          = dic['tol']
    verbose      = dic['verbose']

    begin = rank * (npairs // size)
    end   = begin + (npairs // size) if rank != size - 1 else npairs

    my_pairs = alpha_lambda_pairs[begin:end]
    print('My rank is {} out of {}, I must compute {}'.format(rank, size, my_pairs))

    for (a, l, type) in my_pairs:
        overlaps = get_average_overlaps(a, l, sigma, experimental, ntrials, d, data_model, th_method, mc_samples, tol, verbose)
        if verbose:
            print(overlaps)

        ret_dic = {'overlaps' : overlaps.__dict__}
        config  = {
            'alpha' : a,
            'lambda': l,
            'sigma' : sigma,
            'type'  : type,
        }

        save_folder = dic['save_folder']
        save_file   = dic['save_file']
        if save_folder[-1] != '/':
            save_folder += '/'
    
    # no need to send to rank 0 because each one will save to its own file
        method = 'EXP'
        if not experimental:
            if th_method == ExactIterationMethod.MC:
                method = 'M-C'
            else:
                method = 'INT'
        
        datalogging.save_overlaps_in_csv(a, sigma, l, overlaps, method, constants.DEFAULT_OVERLAPS_FOLDER + constants.OVERLAPS_CSV_FILE_NAME)
        datalogging.save_config_and_result(config, ret_dic, save_folder + datalogging.get_time_default_name(save_file))
