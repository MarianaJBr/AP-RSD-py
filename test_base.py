from __future__ import print_function
import sys, os
import numpy as np

from chi2xipolesAP import chi2xipoles

file_path = os.path.dirname(os.path.abspath(__file__))
print(file_path)
sys.path.insert(0, os.path.join(file_path, '..'))

BFVfile = os.path.join(file_path,
                         './output-from-scripts/BFV-20170623-190056.txt'
                        )
[chi2min, abest, ebest] = np.loadtxt(BFVfile)

def test_chi2min():
    '''
    test for values of chi2(alpha, epsilon) function
    :param a: 
    :param e: 
    :return: 
    '''
    assert chi2xipoles(1.05, -0.1) == chi2min