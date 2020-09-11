import numpy as np
from pkmu_frompk import pkclassdata
from xirpoles_frompkell import vxiell_ru
from AP_xi_ell import *

# Now we want to make a script to find the values of monopole and quadrupole using the CLASS pk as input
b, f, sv = 2, 0.7, 1

kvals, pkvals = pkclassdata[:, 0], pkclassdata[:, 1]
nk = np.size(kvals)
nr = nk
kmin, kmax = np.amin(kvals), np.amax(kvals)

rvals = np.linspace(1, 200, nr)

rkvals = kvals * rvals

xi_ell_results = np.array(kvals, rvals, rkvals,
                          vxiell_ru(kmin, kmax, rvals, b, f, sv, 0),
                          vxiell_ru(kmin, kmax, rvals, b, f, sv, 2))

np.savetxt('./xir_model_poles.dat', xi_ell_results)

