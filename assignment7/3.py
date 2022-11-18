import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


if __name__ == '__main__':
    n = 1_000_000
    # Generate u and v, the upper bound 0.75 is found by trail and error
    u = np.random.rand(n)
    v = np.random.rand(n) * 2 / np.e
    r = v / u
    accept = u < np.sqrt(np.exp(-r))
    z = r[accept]

    log = Log()
    log.append('Accept probability is', np.mean(accept))
    log.save('3.txt')

    # Make the figure
    plt.figure()
    x_max = 8
    bins = 20
    # We need to normalize the analytic exponential, so it has the same area under the curve as the histogram
    normalization = x_max / bins * z.size
    x = np.linspace(0, x_max, num=100)
    plt.hist(z, range=(0, x_max), bins=bins, label='Exponential deviates from ROU')
    plt.plot(x, normalization * np.exp(-x), label='Analytic exponential function')
    plt.xlabel('$x$')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('3.pdf')
