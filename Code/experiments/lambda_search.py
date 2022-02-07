# lambda_search.py
# Use : compute the lambda that minimizes the generalisation error ONLY (so only one file to save to)

import csv, json, sys
import os
from time import time
from typing import List, Tuple

sys.path.append('core')
sys.path.append('experiments')
sys.path.append('../Libraries/')
sys.path.append('../Libraries/GCMProject')
# Load module from Benjamin's github (improved version of stateevolution)
improved_se = __import__("state-evolution-erm-logistic")

from mpi4py import MPI
import numpy as np
from sacred import Experiment
import scipy.optimize as optimize

import constants
import data
import datalogging
import gcm_erm

# MPI
comm = MPI.COMM_WORLD
size : int = comm.Get_size()
rank : int = comm.Get_rank()
# sacred 
ex = Experiment('lambda_search')

def error_and_loss_lamb(lamb, d, alpha, sigma, theta):
    res_gcm = gcm_erm.compute_se(d, alpha, sigma, lamb, theta)
    return res_gcm['test_error'], res_gcm['test_loss']

def error_lamb(lamb, d, alpha, sigma, theta, compute_loss):
    """
    Function to minimize for the test ERROR (not test loss)
    """
    error, loss = error_and_loss_lamb(lamb, d, alpha, sigma, theta)
    return loss if compute_loss else error

def calibration_lamb(lamb, d, alpha, sigma, theta):
    res_gcm = gcm_erm.compute_se(d, alpha, sigma, lamb, theta)
    return res_gcm['calibration']

@ex.config
def config():
    # noise of the data / strength
    sigma      = 0.

    alpha_min = .1
    alpha_max = 5.
    nalpha    = 10

    alpha : List[float] = None

    if alpha is None:
        alpha = np.linspace(alpha_min, alpha_max, nalpha).tolist()
    
    # NOTE : We get rid of these params because they may be inconsistent with alpha
    del alpha_min, alpha_max, nalpha

    # === ACTUAL HYPERPARAMETERS OF THE SCRIPT
    # bounds for searching the optimal lambda
    lambda_min = 1e-4
    lambda_max = 1.
    # can be 'se' (for the loss with state evolution) or 'mc' for monte-carlo
    
    save_folder = constants.DEFAULT_LAMBDA_SEARCH_FOLDER
    save_file   = constants.DEFAULT_LAMBDA_SEARCH_FILE_NAME

    # If True, does not do anything (does not save either) and simply tests that MPI is working
    smoke_test  = False
    # If False -> gen. loss, if True -> gen. error
    compute_loss=False
    # If True, overrides the other parameters and minimizes the calibration at a fixed level
    # in that case, code_version MUST be 'gcm'
    compute_calibration = False
    # can be 'ben' for Benjamin's code or 'gcm' for GCMProject (both are in Libraries folder)
    code_version = 'ben'

    dic = dict(locals())

@ex.automain
def main(dic):
    alpha        = dic['alpha']
    sigma        = dic['sigma']
    lambda_min   = dic['lambda_min']
    lambda_max   = dic['lambda_max']
    smoke_test   = dic['smoke_test']
    compute_loss = dic['compute_loss']
    compute_calibration = dic['compute_calibration']
    code_version = dic['code_version']

    nalpha       = len(alpha)
    max_steps    = 3000
    precision    = 1e-9
    damping_coef = 0.9
    model        = 'logistic'

    send_lambda = []

    begin = rank * (nalpha // size)
    end   = begin + (nalpha // size) if rank != size - 1 else nalpha
    
    my_alpha = alpha[begin:end]
    if smoke_test:
        print('My rank is {} and alpha is {}'.format(rank, my_alpha))

    ret_dic = {'alpha' : [], 'lambda_error' : []}

    for index, a in enumerate(my_alpha):
        if smoke_test:
            lambd_opt = begin + index
        elif compute_calibration:
            if not code_version == 'gcm':
                raise Exception('For calibration, code_version MUST be gcm')
            d       = 1000
            theta   = data.iid_teacher(d = 2000)
            result = optimize.minimize_scalar(lambda l : calibration_lamb(l, d, a, sigma, theta), 
                                        bounds=[(lambda_min, lambda_max)])
            lambd_opt = result.x

        else:
            if code_version == 'ben':
                result = improved_se.optimal_lambda(alpha=a,
                                    model=model,
                        sigma=sigma,
                        max_steps=max_steps,
                        use_adapt_damping=False,
                        precision=precision,
                        damping_coef=damping_coef,
                        load_init=False,
                        verbose=False,
                        lambda_min=lambda_min,
                        lambda_max=lambda_max,
                        compute_loss=compute_loss
                        )
                lambd_opt, eg_opt, lossg_opt = result['lambd_opt'], result['eg_opt'], result['lossg_opt']
            elif code_version == 'gcm':
                d       = 1000
                theta   = data.iid_teacher(d = 2000)
                result = optimize.minimize_scalar(lambda l : error_lamb(l, d, a, sigma, theta, compute_loss), 
                                            bounds=[(lambda_min, lambda_max)])
                lambd_opt = result.x
                eg_opt, lossg_opt = error_and_loss_lamb(lambd_opt, d, a, sigma, theta)
            else:
                raise NotImplementedError
            # save lambda in .csv file in addition to .json file specific to the run
            if not compute_loss:
                datalogging.save_lambda_in_csv(a, sigma, lambd_opt, constants.DEFAULT_LAMBDA_SEARCH_FOLDER + constants.LAMBDA_SEARCH_CSV_FILE_NAME, eg=eg_opt, lossg=lossg_opt, method=code_version)
            else:
                datalogging.save_lambda_in_csv(a, sigma, lambd_opt, constants.DEFAULT_LAMBDA_SEARCH_FOLDER + constants.LAMBDA_SEARCH_LOSS_CSV_FILE_NAME, eg=eg_opt, lossg=lossg_opt, method=code_version)

        ret_dic['alpha'].append(a)
        ret_dic['lambda_error'].append(lambd_opt)

    ret_dic['lambda_error'] = comm.gather(ret_dic['lambda_error'], root=0)
    ret_dic['alpha'] = comm.gather(ret_dic['alpha'], root=0)

    if rank == 0:
        ret_dic['lambda_error'] = [x for l in ret_dic['lambda_error'] for x in l]
        ret_dic['alpha']        = [x for l in ret_dic['alpha'] for x in l]

    if smoke_test and rank == 0:
        print('Alpha : ', ret_dic['alpha'])
        print('Lambda error : ', ret_dic['lambda_error'])

    if not smoke_test:
        save_folder = dic['save_folder']
        if save_folder[-1] != '/':
            save_folder += '/'
        
        save = save_folder + datalogging.get_time_default_name(dic['save_file'])
        config = {
            'alpha' : alpha,
            'model' : model,
            'sigma' : sigma,
            'compute_loss' : compute_loss,
            'precision' : precision,
            'lambda_min'  : lambda_min,
            'lambda_max' : lambda_max
        }

        datalogging.save_config_and_result(config, ret_dic, save)