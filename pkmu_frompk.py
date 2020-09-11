# # This script contains the most basic model for P (k, mu) using Kaiser and  FoG terms
# # Mariana Jaber
# # feb 2017

import os
from math import exp, log, floor

import numpy as np
from numba import cfunc
from numba import jit
from numba import vectorize
from scipy import interpolate, integrate

QUAD_EPSABS = 1.49e-8
inf = np.inf
quad = integrate.quad

# pkoutputpath = '/home/mjb/codigos/class/class_v2.0.1/output/'

# Original
# pkoutputpath = './../output-class/'
# pkclassdata = np.loadtxt(pkoutputpath + 'z55_pk.dat', dtype=float)

# Modified by O. A. R. L.
current_dir = os.path.dirname(__file__)
# pkoutputpath = os.path.join(current_dir, '..', 'output-class', 'z55_pk.dat')
pkoutputpath = os.path.join(current_dir, '..', 'output-class', 'mjb_mockpk.dat')
pkclassdata = np.loadtxt(pkoutputpath, dtype=float)
#

def pkclass_fn(kvals, pkvals, kind='linear'):
    '''
    Given the output of class in format (k, pk) values we interpolate the function P(k)
    :param kvals: array of k values
    :param pkvals: corresponding array of pk values
    :param kind:
    :return: the interpolating function found with these kvals and pkvals
    '''
    pkfun = interpolate.interp1d(kvals, pkvals, kind=kind)
    return pkfun


# now we use the interpolating function and apply it to an array of k's and pk's from class
pklin = pkclass_fn(pkclassdata[:, 0], pkclassdata[:, 1])
kmin = np.amin(pkclassdata[:, 0])
kmax = np.amax(pkclassdata[:, 0])
ks_unif = np.logspace(np.log(kmin), np.log(kmax), pkclassdata.shape[0],
                      base=np.e)
# print(ks_unif.shape[0])
pks_unif = pklin(ks_unif)


@vectorize(nopython=True)
def pklin_nb(k):
    sks = ks_unif.shape[0]
    lk = log(k)
    lkmin = log(kmin)
    lkmax = log(kmax)

    if lk == lkmax:
        return pks_unif[-1]

    rt = (lk - lkmin) / (lkmax - lkmin)
    idx = floor(rt * (sks - 1))
    # print(lk, lkmin, lkmax, rt, idx)

    ks_idx = ks_unif[idx]
    ks_idx_n = ks_unif[idx + 1]
    pks_idx = pks_unif[idx]
    pks_idx_n = pks_unif[idx + 1]

    # Linear interpolation
    slope = (pks_idx_n - pks_idx) / (ks_idx_n - ks_idx)
    return slope * (k - ks_idx) + pks_idx


# instead of dealing with every element independently we create a coordinate matrix k, mu

def kmumatrix(kvals, nmuvals):
    '''
    :param kvals:
    :param mu:
    :return: coordinate matrix
    '''
    muvals = np.linspace(0, 1, nmuvals)
    K, MU = np.meshgrid(kvals, muvals, index='ij')
    return K, MU


# @jit
def pkmu(k, mu, b, f, sigmav):
    '''
    Given the linear power spectrum P(k) we build the simple model for P(k,mu)
    :param k: k values
    :param mu: mu = cos(theta) values
    :param pk: p_lin(k) values
    :param b: bias factor
    :param f: linear growth value
    :param sigmav: FoG velocity dispersion
    :return: P(k, mu)
    '''
    beta = f / b
    xfog2 = (k * mu * f * sigmav) ** 2
    # lather on we can make a switch to change from exponential to lorentzian in FoG
    fog = exp(-xfog2)
    kaiser = b ** 2 * (1 + beta ** 2 * mu ** 2) ** 2
    pk = pklin(k)
    # pk = pklin_nb(k)
    return pk * kaiser * fog


# @jit(nopython=True)
def pklintegrand(mup, k, b, f, sigmav, order):
    '''integral will be over first variable always'''
    if order == 0:
        Ll = 1.
    elif order == 2:
        Ll = (1. / 2.) * (3. * mup ** 2 - 1.)
    elif order == 4:
        Ll = (1. / 8.) * (35. * mup ** 4 - 30. * mup ** 2 + 3.)
    else:
        raise ValueError('Invalid order for Legendre polynomial')
    return pkmu(k, mup, b, f, sigmav) * Ll


# jit_pklintegrand = jit(pklintegrand)
# nb_pklintegrand = cfunc('f8(f8, f8, f8, f8, f8, f8)')(pklintegrand)


# @jit
def pkell(k, b, f, sigmav, order):
    """
    :param k:
    :param order: l= 0, 2, or 4
    :return:the l=order multipole of Powerspectrum  given P(k,mu)
    """
    c = 2 * order + 1
    fn = pklintegrand
    # fn = jit_pklintegrand
    # fn = nb_pklintegrand.ctypes
    int, err = integrate.quad(fn, 0, 1,
                              epsabs=QUAD_EPSABS,
                              args=(k, b, f, sigmav, order))
    return c * int


vpkell = np.vectorize(pkell, excluded=['b', 'f', 'sigmav', 'order'])
