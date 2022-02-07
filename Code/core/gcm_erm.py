'''
gcm_erm.py
File with the code for state evolution for ERM
The file is named like this because it comes from the repo 'IDEPHICS/GCMProject' 
'''

import sys
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Use GCMProject => Empirical Risk Minimization
"""

# dirty code : import by adding parent folder
sys.path.append('../Libraries/GCMProject')
# L2 classification task = L2 regression + sign function for prediction
from state_evolution.models.l2_classification import L2Classification
from state_evolution.models.logistic_regression import LogisticRegression
from state_evolution.models.ridge_regression import RidgeRegression
from state_evolution.algorithms.state_evolution import StateEvolution # Standard SP iteration
from state_evolution.data_models.custom import Custom # Custom data model. You input the covariances

def compute_se(d : int, alpha : float, sig : float, lamb : float, theta : List[float] = None, task : str ='logistic') -> dict:
    # teacher and student have same input dimension
    p = d

    Psi = np.identity(d)
    Omega = np.identity(d)
    Phi = np.identity(d)

    # parameter sampled with identity covariance
    # if argument not provided then sample one at random
    if theta is None:
        theta = np.random.normal(0, 1, p)

    data_model = Custom(teacher_teacher_cov = Psi,
                        student_student_cov = Omega,
                        teacher_student_cov = Phi,
                        teacher_weights = theta)

    if task == 'classification':
        # similar to ridge regression
        task_object = L2Classification(sample_complexity = alpha,
    # regularization should be 1 here because prior has variance 1, right ? 
                        regularisation = lamb,
                        data_model = data_model)
    elif task == 'logistic':
    # NOTE : With this function, the data_model by default is the probit model
        task_object = LogisticRegression(
    # NOTE : In Logistic regression, we need to provide the VARIANCE
                        sig**2,
                        sample_complexity = alpha, 
                        regularisation = lamb,
                        data_model = data_model)
    elif task == 'ridge':
        task_object = RidgeRegression(sample_complexity = alpha, 
                        regularisation=lamb,
                        data_model = data_model)
    else:
        print('Wrong task : ', task)
        raise NotImplementedError

    sp = StateEvolution(model = task_object,
                        initialisation = 'uninformed',
                        tolerance = 1e-8,
                        damping = 0.5,
                        verbose = False,
                        max_steps = 100000)

    sp.iterate()
    dic = sp.get_info()
    # norme carree du vrai = 1 car covariance est = identité 
    rho = 1.

    dic['rho'] = rho
    dic['mse'] = rho + dic['overlaps']['self_overlap'] - 2.*dic['overlaps']['teacher_student']
    if task == 'logistic':
        V, q, m = dic['overlaps']['variance'], dic['overlaps']['self_overlap'], dic['overlaps']['teacher_student']
        dic['hatV'], dic['hatq'], dic['hatm'] = task_object._update_hatoverlaps(V, q, m)
    return dic
