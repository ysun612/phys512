import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


if __name__ == '__main__':
    n = 1_000_000
    # Scale random number to [0.5, 1), so that the Lorentzian is for x>0
    r = np.random.rand(n) / 2 + 0.5
    # Generate the Lorentzian
    y = np.tan(np.pi * (r - 0.5))

    # Accept probability
    prob = np.exp(-y) / (1 / (1 + y ** 2))
    # Accept with the given accept probability
    accept = np.random.rand(n) < prob
    z = y[accept]

    log = Log()
    log.append('Accept probability is', np.mean(accept))
    log.save('2.txt')

    # Make the figure
    plt.figure()
    x_max = 8
    bins = 20
    # We need to normalize the analytic exponential, so it has the same area under the curve as the histogram
    normalization = x_max / bins * z.size
    x = np.linspace(0, x_max, num=100)
    plt.hist(z, range=(0, x_max), bins=bins, label='Exponential deviates from rejection')
    plt.plot(x, normalization * np.exp(-x), label='Analytic exponential function')
    plt.xlabel('$x$')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2.pdf')
