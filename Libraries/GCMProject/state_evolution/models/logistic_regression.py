from math import erfc
import numpy as np
import scipy.stats as stats
import scipy.integrate
from .base_model import Model
from ..auxiliary.logistic_integrals import integrate_for_mhat, integrate_for_Vhat, integrate_for_Qhat, traning_error_logistic

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

class LogisticRegression(Model):
    '''
    Implements updates for logistic regression task.
    See base_model for details on modules.
    '''
    def __init__(self, Delta = 0., *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Delta = Sigma**2 
        self.Delta = Delta
        self.data_model = data_model

    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        # should not be affected by the noise level
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                                           mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                                          (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat / np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit/(self.lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU * self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/(self.lamb + Vhat * self.data_model.spec_Omega))

        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        Vstar = self.data_model.rho - m**2/q

        Im = integrate_for_mhat(m, q, V, Vstar + self.Delta)
        Iv = integrate_for_Vhat(m, q, V, Vstar + self.Delta)
        Iq = integrate_for_Qhat(m, q, V, Vstar + self.Delta)

        mhat = self.alpha/np.sqrt(self.data_model.gamma) * Im/V
        Vhat = self.alpha * ((1/V) - (1/V**2) * Iv)
        qhat = self.alpha * Iq/V**2

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)

    def get_test_error(self, q, m):
        return np.arccos(m/np.sqrt(q * (self.data_model.rho + self.Delta)))/np.pi

    def get_test_error_0(self, q, m):
        # Test error if the test data is NOT noisy, useful to minimize the test error more efficiently 
        return np.arccos(m/np.sqrt(q * self.data_model.rho))/np.pi

    def get_test_loss(self, q, m):
        Sigma = np.array([
            [self.data_model.rho + self.Delta, m],
            [m, q]
        ])

        def loss_integrand(lf_teacher, lf_erm):
            return np.log(1. + np.exp(- np.sign(lf_teacher) * lf_erm)) * stats.multivariate_normal.pdf([lf_teacher, lf_erm], mean=np.zeros(2), cov=Sigma)
        
        ranges = [(-10.0, 10.0), (-10.0, 10.0)]
        lossg_mle = scipy.integrate.nquad(loss_integrand, ranges)[0]
        return lossg_mle
    
    def get_calibration(self, q, m, p=0.75):
        inv_p = sigmoid_inv(p)
        rho   = self.data_model.rho
        return p - 0.5 * erfc(- (m / q * inv_p) / np.sqrt(2*(rho - m**2 / q + self.Delta)))

    def get_train_loss(self, V, q, m):
        Vstar = self.data_model.rho - m**2/q
        return traning_error_logistic(m, q, V, Vstar + self.Delta)
