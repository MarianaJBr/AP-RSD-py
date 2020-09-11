# ==============================================================
# this script contains the definition of
# sperical bessel function j0, j1 and j2 of zeroth order
# using the recurrence relation 14.173 of Arfken & Weber 7th ed:
# jn(x) = (-1)**n *x**n (1/x d/dx)**n (sinx/x)
# in terms of trig functions to speed up the calculation
# errors are of order 10**-11 and smaller
# Mariana Jaber // Feb 2017
# ==============================================================


from math import sin, cos
from numba import jit


@jit(nopython=True)
def sincf(x):
    '''
    sinc function
    '''
    if x == 0:
        return 1.0
    return sin(x) / x

@jit(nopython=True)
def j0bess(x):
    '''
    spherical bessel of order zero
    j0(x)= sinc(x)
    '''
    if x == 0:
        j0 = 1
    else:
        j0 = sincf(x)
    return j0

@jit(nopython=True)
def j2bess(x):
    '''
    2nd order spherical bessel function
    given as j2(x) = x**2(1/x)(d/dx)((1/x)(dj0(x)/dx))
    with j0(x)  = sinc(x)
    j2(x) = 3*sinc(x)/x**2-2*cos(x)/x**2-sinc(x)
    '''
    return 3 * sincf(x) / x ** 2 - 2 * cos(x) / x ** 2 - sincf(x)

@jit(nopython=True)
def j4bess(x):
    '''
    4th order spherical bessel function using
    the prescription:
    j4(x) = x**3 (d/dx)(1/x d/dx)(j2(x)/x**2)
    '''
    c1 = 5 * x * (2 * x ** 2 - 21)
    c2 = x ** 4 - 37 * x ** 2 + 33
    c3 = -8 * x ** 3 + 72 * x
    return c1 * cos(x) / x ** 5 + c2 * sin(x) / x ** 5 + c3 * sincf(x) / x ** 5

