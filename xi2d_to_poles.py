# this script contains the functions to find the multipoles of 2D-Correl Func
# xi(r, mu) --> xi_l(r) by integrating over Legendre polynomials
# it takes the ouptut from GSRSD for xi(r, mu) and project it into the Legendre polynomials base
# to obtain xi_l(r) with l=0, 2, 4.
# Mariana Jaber. feb 2017.

import os
import numpy as np
from scipy import integrate
from scipy.interpolate import interpolate

QUAD_EPSABS = 1.49e-8

current_dir = os.path.dirname(__file__)
# pkoutputpath = os.path.join(current_dir, '..', 'output-class', 'z55_pk.dat')

gsrdoutput = os.path.join(current_dir, '..', 'output-class', 'xi_s_mock.txt_2d')
xi2Ddata = np.loadtxt(gsrdoutput, dtype=float)
# I know I created the output for GSRD with 50 bins for r.
# The shape of data is (3200, 3) which tells me I have 64 values of mu for every r.
# So I will reshape the data into (rsize, musize, 3)
# To obtain coordinate matrices for r, mu, xi needed for 2d interpolation

xi2new = xi2Ddata.reshape(50, 64, 3)

rdatamatrix = xi2new[:, :, 0]
mudatamatrix = xi2new[:, :, 1]
xidatamatrix = xi2new[:, :, 2]

# and the corresponding 1D arrays for r and mu:
rdata1D = rdatamatrix[:, 0]
mudata1D = mudatamatrix[0, :]


# first we need to interpolate the output from GSRSD to create the function xi(r, mu)
def xi2d_func(rvals, muvals, xi2dvals):
    '''
    This function interpolates the values of xi2d over the values of r and mu as returned by GSRSD
    :param rvals: configuration space coordinates. 's' in GSRSD output since it is in redshift space. 1D array
    :param muvals: mu = cos(theta) values. mu in (0, 1). 1D array
    :param xi2dvals: xi values corresponding to each (r, mu) pair. (rsize, musize) matrix
    :return: interpolating functions for xi over the (r, mu)  values
    '''
    xi2d = interpolate.RectBivariateSpline(rvals, muvals, xi2dvals)

    return xi2d


# now we use the interpolating function and apply it to an array of r's, mu's and xi's.
# it can be the same rdatamatrix, mudatamatrix from GSRSD or different arrays

xi2Dinterpoladora = interpolate.RectBivariateSpline(rdata1D,
                                         mudata1D,
                                         xidatamatrix)

#this returns an array with values for xi
xi2d=xi2Dinterpoladora(rdata1D,mudata1D)



# next we proceed to define the integrand for the xi_l(r) function
def xi_poles_integrand(mup, r1d, order):
    '''
    The definition for xi(r, mu)*Legendre_l(mu)*dmu integrand function
    :param xi: value of the 2D Correlation Function as given by GSRD (or other)
    :param mup: INTEGRATION VARIABLE. muprime = cosine of theta values (from 0 to 1 in GSRD output)
    :param r: configuration space coordinate
    :param order: order for the legendre polynomial order l = 0, 2, 4
    :return: xi(r, mu') * Legendre_l(mu')
    '''
    if order == 0:
        Ll = 1.
    elif order == 2:
        Ll = (1. / 2.) * (3. * mup ** 2 - 1.)
    elif order == 4:
        Ll = (1. / 8.) * (35. * mup ** 4 - 30. * mup ** 2 + 3.)
    else:
        raise ValueError('Invalid order for Legendre polynomial')
    xi = xi2Dinterpoladora(r1d, mup)
    return float(xi[0, 0]) * Ll


def xi_poles(order):
    '''
    Integral of xi(r, mu')*L_l(mu') over all mu' to obtain xi_l(r)
    :param order: l= 0, 2, 4
    :return: xi_l(r)
    '''
    coeff = 2 * order + 1
    fn = xi_poles_integrand
    int, error = integrate.quad(fn, 0, 1,
                                epsabs = 1.49e-8,
                                args = (r1d, order))
    return coeff * int


# Finally we vectorize the result to apply it to
vxi_poles=np.vectorize(xi_poles, excluded=['order'])


