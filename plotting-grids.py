## sequential plotting
import  numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import contourf, show

from chi2xipolesAP import chi2xipoles



def sequantialplotchi2grid(ngrid):
    '''
    First we vectorize the chi2 function over alpha, epsilon
    so it can be fed with matrix /arrays like elements

    Then we create a grid of ngrid x ngrid for alpha-epsilon

    We map the vectorize function over the grid and save the output

    :param ngrid: number of elements for alpha and epsilon
    :return: chi2 evaluated in the grid for every of the nxn points
    '''
    BFV_file = np.loadtxt('./output-from-scripts/BFV-20170305-235140.txt')
    (chi2b, alphab, epsilonb) = BFV_file

    vchi2_xipoles = np.vectorize(chi2xipoles)

    alphas = np.linspace(0.001, 3, ngrid)
    epsilones = np.linspace(-0.99, 1.2, ngrid)

    Alpha, Epsilon = np.meshgrid(alphas, epsilones, indexing='ij')
    chi2_grid = vchi2_xipoles(Alpha, Epsilon)

    n = 32
    m = 2
    nu = n - m

    #d1s, d2s, d3s = 33.1211, 44.2248, 56.0425
    d1s, d2s, d3s = 2.30, 6.17, 11.8
    sigma1, sigma2, sigma3 = chi2b+d1s,  chi2b+d2s,  chi2b+d3s
    levels = [sigma1, sigma2, sigma3]

    chi2contourplot = contourf(Alpha, Epsilon, chi2_grid, levels=levels)

    chi2contours = contourf(Alpha, Epsilon, chi2_grid, 1000)

    ngridsize = str(ngrid)
    filename1 = './../output-figs/contour-n_' + ngridsize+'.pdf'
    filename2 = './../output-figs/contour-n_' + ngridsize+'-2.pdf'
    pyplot.xscale('log')
    pyplot.ylabel('epsilon')
    pyplot.xlabel('alpha')
    pyplot.colorbar()
    pyplot.savefig(filename1, dpi=300, bbox_inches='tight')
    pyplot.savefig(filename2, dpi=300, bbox_inches='tight')

    show()
    return chi2contourplot

#sequantialplotchi2grid(10)
