# # This script calculates the multipoles of correlation function \xi_l(r) from
# # the multipoles of power spectrum: P_l(k)

# # Mariana Jaber
# # feb 2017

import math
import numpy as np
from scipy import integrate
from pkmu_frompk import pkell
from sphbessel import *

QUAD_EPSABS = 1.49e-8
inf = np.inf


# quad = integrate.quad


def xiellu_integrand(u, r, b, f, sigmav, order):
    eu = math.exp(u)
    kr = r * eu
    if order == 0:
        jlkr = j0bess(kr)
    elif order == 2:
        jlkr = j2bess(kr)
    elif order == 4:
        jlkr = j4bess(kr)
    else:
        raise ValueError('Invalid order for Bessel function')
    coeffu = math.exp(3 * u) / (2 * np.pi ** 2)
    pkl = pkell(eu, b, f, sigmav, order)
    return coeffu * pkl * jlkr


v_xiellu_integrand = np.vectorize(xiellu_integrand,
                                  excluded=['b', 'f', 'sigmav', 'order'])


def xiellu_r(kmin, kmax, r, b, f, sigmav, order):
    '''
    Integral over u=ln(k)
    :param kmin: min value of k range in P_lin(k)
    :param kmax: max value of k range in P_lin(k)
    :param r: r values for Xi(r)
    :param b: bias factor
    :param f: linear growth factor
    :param sigmav: FoG velocity dispersion factor
    :param order: L = 0, 2, 4 order for the multipole
    :return: xi_l(r) with l= order
    '''
    umin = math.log(kmin)
    umax = math.log(kmax)
    if (order == 0 or order == 4):
        itol = 1.
    elif (order == 2):
        itol = -1.
    else:
        raise ValueError('Invalid order for Legendre polynomial')
    int, err = integrate.quad(xiellu_integrand, umin, umax,
                              epsabs=QUAD_EPSABS,
                              args=(r, b, f, sigmav, order))
    return itol * int


vxiell_ru = np.vectorize(xiellu_r,
                         excluded=['kmin', 'kmax', 'b', 'f', 'sigmav', 'order'])


# ============ version without logarithmic integrand ==================================#

def xiell_integrand(k, r, b, f, sigmav, order):
    '''
    :param k: integration variable
    :param r: config space coordinate
    :param b: bias factor
    :param f: linear growth
    :param sigmav: FoG velocity dispersion
    :param order: l = 0, 2
    :return: (k**2/2pi**2) P_l(k)j_l(kr)
    '''
    kr = k * r
    if order == 0:
        jlkr = j0bess(kr)
    elif order == 2:
        jlkr = j2bess(kr)
    elif order == 4:
        jlkr = j4bess(kr)
    else:
        raise ValueError('Invalid order for Bessel function')
    # jlkr = sjn(order, kr, derivative=False)
    coeffk = k ** 2 / (2 * np.pi ** 2)
    pkl = pkell(k, b, f, sigmav, order)
    return coeffk * pkl * jlkr


def xiell_r(kmin, kmax, r, b, f, sigmav, order):
    if (order == 0 or order == 4):
        itol = 1.
    elif (order == 2):
        itol = -1.
    else:
        raise ValueError('Invalid order for Legendre polynomial')
    int, err = integrate.quad(xiell_integrand, kmin, kmax,
                              epsabs=QUAD_EPSABS,
                              args=(r, b, f, sigmav, order))
    return itol * int


vxiell_r = np.vectorize(xiell_r,
                        excluded=['kmin', 'kmax', 'b', 'f', 'sigmav', 'order'])

# =================================================================================== #
