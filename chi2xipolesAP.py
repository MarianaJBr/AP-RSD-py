import numpy as np
import datetime
import h5py
from matplotlib import pyplot
from matplotlib.pyplot import contourf, show

from AP_xi_ell import r36t156mock, r36t156model, xi0mock, xi2mock, vxi_poles

xi0mockcut = xi0mock[4:20]
xi2mockcut = xi2mock[4:20]
covmat = np.loadtxt('./../data-files/cov36-156.txt')


def chi2xipoles(alpha, epsilon):
    '''
    Calculates the chi squared function for alpha and epsilon
    '''

    r = r36t156mock

    invcov = np.linalg.inv(covmat)

    xi0model = vxi_poles(r, alpha, epsilon, 0)
    xi2model = vxi_poles(r, alpha, epsilon, 2)

    xi0data = xi0mockcut
    xi2data = xi2mockcut

    ymodel = np.concatenate((xi0model, xi2model), axis=0)
    ydata = np.concatenate((xi0mockcut, xi2mockcut), axis=0)

    yxipoles = ymodel - ydata

    return np.dot(yxipoles, np.dot(invcov, yxipoles))


# now we need to vectorize it to map it unto a grid of Alpha, Epsilon values

def loglike(alpha, epsilon):
    '''
    ~log(Like) = -chi2/2
    :param alpha: 
    :param epsilon:  
    '''
    return  - chi2xipoles(alpha, epsilon) / 2


def proba(alpha, epsilon):
    '''
    probability function (modulus the normalization constant, 
    defined by \int p(x)dx=1
                                )
    :param alpha: 
    :param epsilon: 
    :return:  p(alpha, epsilon) ~ exp(-chi2(alpha, epsilon)/2)
    '''
    arg = - chi2xipoles(alpha, epsilon) / 2
    return np.exp(arg)



############################################################################
# ## sequential processing: # # #
HEADER = ''' chi2(alpha, epsilon) values
when applied to a grid of size n x n
    alphas = np.linspace(0.5, 1.5, ngrid)
    epsilones = np.linspace(-0.5, 0.5, ngrid)
'''


def chi2sequentialgrid(ngrid):
    '''
    First we vectorize the chi2 function over alpha, epsilon
    so it can be fed with matrix /arrays like elements

    Then we create a grid of ngrid x ngrid for alpha-epsilon

    We map the vectorize function over the grid and save the output

    :param ngrid: number of elements for alpha and epsilon
    :return: chi2 evaluated in the grid for every of the nxn points
    '''
    vchi2_xipoles = np.vectorize(chi2xipoles)

    alphas = np.linspace(0.5, 1.5, ngrid)
    epsilones = np.linspace(-0.5, 0.5, ngrid)

    Alpha, Epsilon = np.meshgrid(alphas, epsilones, indexing='ij')

    chi2_grid = vchi2_xipoles(Alpha, Epsilon)

    full_result = np.array(
        [Alpha, Epsilon, chi2_grid]
    )
    # name for the file
    ngridsize = str(ngrid)
    filename = './output-from-scripts/chi2grid-n_' + ngridsize
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d-%H%M%S')

    filetxt = filename + '-{:s}.txt'.format(
        date_str)
    np.savetxt(filetxt, chi2_grid,
               header=HEADER)

    return chi2_grid



    # now = datetime.datetime.now()
    # date_str = now.strftime('%Y%m%d-%H%M%S')
    #
    # filename = './output-from-scripts/grid-{:s}.h5'.format(
    #     date_str)
    # with File(filename, 'w') as file:
    #     try:
    #         # Try to delete existent data
    #         del file['/DataGrid']
    #         del file['/Chi2']
    #     except KeyError:
    #         # If data not exists, do nothing
    #         pass
    #
    #     file.attrs['Comment'] = '''
    #             Grid created from the best fit values
    #     contained in the following file:
    #
    #         "./output-from-scripts/BFV-20170305-235140.txt"
    #     '''
    #     # Add data
    #     file['/DataGrid'] = grid_data
    #     file['/Chi2'] = chi2_grid
    #
    #     # Save data to file
    #     file.flush()

###




# chi2grid(10)
