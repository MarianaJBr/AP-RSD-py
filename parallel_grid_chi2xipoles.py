from __future__ import print_function
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from h5py import File
from parallel_chi2 import parallel_chi2any
from chi2xipolesAP import chi2xipoles


#BFV_file = np.loadtxt('./output-from-scripts/BFV-20170622-173836.txt')
#(chi2b, alphab, epsilonb) = BFV_file


def chi2_kernel(data):
    alpha, epsilon = data
    chi = chi2xipoles(alpha, epsilon)
    print("Completed chi2 for", data)
    print("Value:", chi)
    return chi


def exec_chi2xipoles():
    ngridsize = 50
    a_values = ngridsize
    e_values = ngridsize

    # alphas = np.linspace(0.01, 3, ngrid)
    # epsilones = np.linspace(-0.9, 1.1, ngrid)
    a_range = np.linspace(0.9, 1.1, a_values)
    e_range = np.linspace(-0.2, 0.2, e_values)


    grid_data = np.zeros((a_values, e_values, 2))

    grid_data[:, :, 0] = a_range[:, np.newaxis]
    grid_data[:, :, 1] = e_range[np.newaxis, :]

    flattened_grid = grid_data.reshape(a_values*e_values,2)

    #Execute parallel routine
    chi2_data = parallel_chi2any(chi2_kernel, flattened_grid)

    chi2data = chi2_data.reshape(a_values,e_values,1)

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d-%H%M')
    ngs = str(ngridsize)
    file_name = './output-from-scripts/grid-n_'+ngs+'-{:s}.h5'
    filename = file_name.format(
        date_str)
    with File(filename, 'w') as file:

        try:
            # Try to delete existent data
            del file['/DataGrid']
            del file['/Chi2']
        except KeyError:
            # If data not exists, do nothing
            pass

        file.attrs['Comment'] = '''
        Grid created from the best fit values
        contained in the following file:
        "./output-from-scripts/BFV-20170622-173836.txt"
        With
        a_values = e_values = n_gridsize = 100
        a_range = np.linspace(0.8, 1.2, a_values)
        e_range = np.linspace(-0.2, 0.2, e_values)
        '''
        # Add data
        file['/DataGrid'] = grid_data
        file['/Chi2'] = chi2data

        # Save data to file
        file.flush()

        print(chi2data)


# Correct indentation level.
if __name__ == '__main__':
    exec_chi2xipoles()

