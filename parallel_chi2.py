# -*- encoding: utf-8 -*-
from multiprocessing.pool import Pool

import numpy as np
import progressbar
from progressbar import Bar, Percentage, ProgressBar, ETA

# parallel implementation of chi2 function
def parallel_chi2any(kernel, data_array, processes=None):
    """Evaluates the chi2BAO function over the data saved in `data_array`
    and distributes the workload among several independent python
    processes.
    """

    # Let's create a pool of processes to execute calculate chi2 in
    # parallel.

    assert hasattr(data_array, 'dtype')

    data_values = data_array.shape[0]

    percent = Percentage()
    eta = ETA()
    bar = Bar(marker=u'☞') # ⚆ ♠ ☙ ☈ ☂ ✂ ✈ ✎ ✑ ✶ ❒ ❚ ❯ ❱ ➠ ➜ ➸ ➽

    progress_bar = ProgressBar(max_value = data_values,
                               widgets=[
                                   '[', percent, ']',
                                   bar,
                                   '(', eta, ')'
                               ])

    # with Pool(processes=processes) as pool:
    #     # python 3 code...

    pool = Pool(processes=processes)
    # The data accepted by the map method must be an iterable, like
    # a list, a tuple or a numpy array. The function is applied over
    # each element of the iterable. The result is another list with
    # the values returned by the function after being evaluated.
    #
    # [a, b, c, ...]  -> [f(a), f(b), f(c), ...]
    #
    # Here we use the imap method, so we need to create a list to
    # gather the results.
    results = []
    pool_imap = pool.imap(kernel, data_array)
    progress = 0
    for result in pool_imap:
        results.append(result)
        progress += 1
        progress_bar.update(progress)

    pool.close()

    return np.array(results)
