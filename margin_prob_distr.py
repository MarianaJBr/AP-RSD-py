from chi2xipolesAP import chi2xipoles, covmat
import numpy as np
from scipy import integrate

# boundsae = [(0.5, 1.5), (-0.5, 0.5)]
# bounds used during minimization
amax = 1.5
amin = 0.5
emax = 0.5
emin = -0.5


def chi2alpha_integrand(epsilon, alpha):
    '''
    integrand for marginalization over alpha so we can calculate chi2(alpha)
    :param epsilon:
    :param alpha:
    :return:  chi2(alpha, epsilon) with correct arguments position for integration
    '''
    chi2 = chi2xipoles(alpha, epsilon)
    return chi2

def chi2epsilon_integrand(alpha, epsilon):
    '''
    integrand for marginalization over alpha so we can calculate chi2(alpha)
    :param alpha:
    :param epsilon:
    :return:  chi2(alpha, epsilon) with correct arguments position for integration
    '''
    chi2 = chi2xipoles(alpha, epsilon)
    return chi2


def chi2alpha(alpha):
    '''
    Gives the chi2 funcion marginalized over epsilon parameter
    :param alpha:
    :return: chi2(alpha)
    '''

    margchi2 = integrate.quadrature(chi2alpha_integrand,
                                    emin, emax,
                                    args=alpha,
                                    vec_func=False
                                    )
    weight = emax - emin
    weighted = margchi2[0] / weight
    return weighted


def chi2epsilon(epsilon):
    '''
    Gives the chi2 function marginalized over alpha parameter
    :param epsilon:
    :return: chi2(epsilon)
    '''
    margchi2 = integrate.quadrature(chi2epsilon_integrand,
                                    amin, amax,
                                    args=epsilon,
                                    vec_func=False)
    weight = amax - amin
    weighted = margchi2[0] / weight
    return weighted


def posteriori(alpha, epsilon):
    '''
    Posteriori probab distribution associated to Gaussian Likelihood for alpha, epsilon, assuming flat priors
    :param alpha:
    :param epsilon:
    :return: P(alpha, epsilon | data ) = Prior(alpha, epsilon) * Likelihood(d|alpha, epsilon)
    '''
    detcov = np.linalg.det(covmat)
    deltaalpha = amax - emin
    deltaepsilon = emax - emin
    priors = (1 / deltaalpha) * (1 / deltaepsilon)
    L0 = 2 * np.pi * np.sqrt(detcov)
    chi2 = chi2xipoles(alpha, epsilon)
    likelihood = np.exp(- chi2 / 2)
    return priors * likelihood / L0

def Palpha_integrand(epsilon, alpha):
    '''
    integrand for marginalization over epsilon values so we can calculate P(alpha|data)
    :param epsilon: values of epsilon
    :param alpha: values of alpha
    :return: exp(-chi2(alpha, epsilon)/2)
    '''
    chi2 = chi2xipoles(alpha, epsilon)
    integrand = np.exp(-chi2 / 2)
    return integrand


def Pepsilon_integrand(alpha, epsilon):
    '''
    integrand for marginalization over alpha values so we can calculate  P(epsilon|data)
    :param epsilon: values of epsilon
    :param alpha: values of alpha
    :return: exp(-chi2(alpha, epsilon)/2)
    '''
    chi2 = chi2xipoles(alpha, epsilon)
    integrand = np.exp(-chi2 / 2)
    return integrand

def probdist_alpha(alpha):
    '''
    Marginalized 1-D probab distribution for alpha assuming flat priors and gaussian likelihood
    :param alpha:
    :return: P(alpha|data)
    '''
    detcov = np.linalg.det(covmat)
    deltaalpha = amax - emin
    deltaepsilon = emax - emin
    priors = (1 / deltaalpha) * (1 / deltaepsilon)
    L0 = 2 * np.pi * np.sqrt(detcov)
    marginalized_alpha = integrate.quadrature(Palpha_integrand,
                                              emin, emax,
                                              args=alpha,
                                              vec_func=False)
    f1 = priors / L0
    f2 = marginalized_alpha[0]
    return f1 * f2


def probdist_epsilon(epsilon):
    '''
    Marginalized 1-D probab distribution for epsilon assuming flat priors and gaussian likelihood
    :param epsilon:
    :return: P(epsilon|data)
    '''
    detcov = np.linalg.det(covmat)
    deltaalpha = amax - emin
    deltaepsilon = emax - emin
    priors = (1 / deltaalpha) * (1 / deltaepsilon)
    L0 = 2 * np.pi * np.sqrt(detcov)
    marginalized_epsilon = integrate.quadrature(Pepsilon_integrand,
                                                amin, amax,
                                                args=epsilon,
                                                vec_func=False)
    f1 = priors / L0
    f2 = marginalized_epsilon[0]
    return f1 / f2


print(chi2alpha(1.1))

