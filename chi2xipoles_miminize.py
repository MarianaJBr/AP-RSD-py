
import datetime

import os, sys
import numpy as np
#from scipy.special import gammainc, gamma
#from scipy.optimize import differential_evolution
from scipy.optimize import differential_evolution, minimize, basinhopping

file_path = os.path.dirname(os.path.abspath(__file__))
print(file_path)
sys.path.insert(0, os.path.join(file_path, '..'))

from chi2xipolesAP import chi2xipoles


HEADER = ''' BFV for alpha, epsilon,
###
differential_evolution(func_to_minimize,
                    bounds=[(0.9, 1.1), (-0.2, 0.2)],
                    tol=1e-4,
                    polish=True)
###
output = chi2, alpha, epsilon
'''



def chi2_minimization():

    bounds = [(0.9, 1.1), (-0.2, 0.2)]

    def func_to_minimize(x):
        """
        :param x:  The point.
        :return:
        """

        alpha, epsilon = x

        val = chi2xipoles(alpha, epsilon)
        print("Point:", x, "Function value:", val)
        print("Function value:", val)
        return val

    t0 = datetime.datetime.now()

    result = differential_evolution(func_to_minimize,
                                    bounds=bounds,
                                    tol=1e-4,
                                    disp=True,
                                    #maxiter=100,
                                    # seed= 1,
                                    polish=True)

    # result = minimize(
    #                 func_to_minimize,
    #                 x0=(1, 0),
    #                 method='Nelder-Mead',
    #                 #bounds=bounds,
    #                 tol=1e-4
    #                 )
    # minimizer_kwargs = {"method": "Nelder-Mead", "tol": "1e-4"}
    # x0 = [1.0, 0.0]
    #
    # result = basinhopping(
    #     func_to_minimize,
    #     x0,
    #     niter=200,
    #     T=1,
    #     stepsize=0.3,
    #     minimizer_kwargs=minimizer_kwargs
    #
    # )

    total = datetime.datetime.now() - t0

    print(result)
    print("Total time", total)

    chi2 = result.fun
    alpha, epsilon = result.x

    # # correspondig likelihood to the minimum of chi2 function
    # # maxlikebao = LikeBAOuncorr(w0, wi, q, zt, REDUCED_H0, omch2)
    # maxlikebao = LikeBAOuncorr(w0, wi, q, zt, hvar, OmegaM)
    #
    # # # goodness of fit as GammaInc(nu/2, chi2min/2)/Gamma(nu/2)
    # # # nu = the number of dof, i.e. nu = data - parameters = N - m
    # ndata = len(REDSHIFTS_RBAO_7new)
    # mparams = len(result.x)
    # nu = ndata - mparams
    # gof = gammainc(nu / 2, chi2 / 2) / gamma(nu / 2)
    # # # BIC criteria defined as BIC = -2ln(LikeMax) + mln(N)
    # bic = -2 * np.log(maxlikebao) + mparams * np.log(ndata)
    # # # AIC criteria defined as AIC = - 2 ln(LikeMax) + 2*m
    # aic = -2 * np.log(maxlikebao) + 2 * mparams
    full_result = np.array(
        [chi2, alpha, epsilon]
    )
    # full_result = np.array(
    #     [chi2, w0, wi, q, zt, hvar, OmegaM, maxlikebao, gof,
    #      bic, aic, mparams])
    #
    # #    [w0, wi, q, zt, REDUCED_H0, omegach2, chi2, maxlikeBAO,
    # #     GoF, bic, aic])

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d-%H%M%S')

    filetxt = './output-from-scripts/BFV-{:s}.txt'.format(
        date_str)
    np.savetxt(filetxt, full_result,
               header=HEADER)


if __name__ == '__main__':
    chi2_minimization()
