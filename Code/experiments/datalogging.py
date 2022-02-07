# logging.py
# Functions to load and save_data

import csv, json, os, sys
from distutils.log import error

sys.path.append('experiments')

from random import randint
import datetime
from typing import Tuple
import numpy as np

import core.overlaps as overlaps
import constants

def convert_to_float(x):
    try:
        return float(x)
    except:
        return x

def get_time_default_name(base_name : str) -> str:
    return datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + '-' + str(randint(0, 1000)) + '-' + base_name

def save_config_and_result(config : dict, results : dict, fname : str) -> None:
    to_save = {'configuration' : config,
               'results'       : results}
    with open(fname, 'w+') as f:
        json.dump(to_save, f)

def load_config_and_result(fname : str) -> Tuple[str, str]:
    with open(fname) as f:
        content = json.load(f)
    return content['configuration'], content['results']

def save_lambda_in_csv(alpha : float, sigma : float, lambd : float, filename : str, eg : float = np.nan, lossg : float = np.nan, method : str = 'n/a'):
    """
    Save ONE lambda in the given file, as a functio nof alpha and sigma
    """
    data = [alpha, sigma, lambd, eg, lossg, method]
    header = ['alpha', 'sigma', 'lambda', 'eg_mle', 'lossg_mle', 'method']
    if not os.path.isfile(filename):
        with open(filename, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(header)   

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def try_loading_lambda_from_csv(sigma : float, filename : str = None, include_errors = False):
    alpha_list, lambda_list = [], []
    error_list, loss_list   = [], []
    if filename == None:
        filename = constants.DEFAULT_LAMBDA_SEARCH_FOLDER + constants.LAMBDA_SEARCH_CSV_FILE_NAME
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            try:
                if '#' in row[-1]:
                    continue
                row = list(map(convert_to_float, row))
                if '#' in row[-1]:
                    continue
                if row[1] == sigma:
                    alpha_list.append(row[0])
                    lambda_list.append(row[2])
                    if include_errors:
                        error_list.append(row[3])
                        loss_list.append(row[4])
            except:
                pass
    if include_errors:
        return alpha_list, lambda_list, error_list, loss_list 
    return alpha_list, lambda_list

# =========

def save_overlaps_in_csv(alpha : float, sigma : float, lambd : float, o : overlaps.Overlaps, method : str, filename : str = None):
    """
    Save ONE lambda in the given file, as a functio nof alpha and sigma
    arguments : 
        - method : method used (experimental, mc or nquad)
    """
    if filename is None:
        filename = constants.DEFAULT_OVERLAPS_FOLDER + constants.OVERLAPS_CSV_FILE_NAME

    data = [method, alpha, lambd, sigma, o.rho, o.qbo, o.qerm, o.mbo, o.m, o.Q]
    header = ['method', 'alpha', 'lambda', 'sigma', 'rho', 'qbo', 'qerm', 'mbo', 'm', 'Q']

    if not os.path.isfile(filename):
        with open(filename, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(header)   

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def try_loading_overlaps_from_csv(alpha : float, sigma : float, lambd : float, filename : str = None, as_object : bool = False):
    # NOTE : Reminder of the order of data : 'method', 'alpha', 'lambda', 'sigma', 'rho', 'qbo', 'qerm', 'mbo', 'm', 'Q']
    if filename is None:
        filename = constants.DEFAULT_OVERLAPS_FOLDER + constants.OVERLAPS_CSV_FILE_NAME

    # search parameters close by tol
    tol = 1e-5
    tab = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            try:
                if row[0] == 'method':
                    continue
                # ignore lines where # in the method
                elif '#' in row[0]:
                    continue
                row = list(map(convert_to_float, row))
                if alpha - tol <= row[1] <= alpha + tol and \
                    lambd - tol <= row[2]<= lambd + tol and \
                    row[3] == sigma:
                    if as_object:
                        alpha, lambd, sigma, rho, qbo, qerm, mbo, m, Q = row[1:]
                        return alpha, lambd, sigma, overlaps.Overlaps(rho = rho, qbo = qbo, qerm = qerm, m = m, Q = Q)
                    return row[1:]
            except:
                pass
    return None