import numpy as np

# A simple logger that acts like print() but can also save outputs to files
import sys
sys.path.append("../")
from utils.logger import Log


def problem1(f, df, x, ds, n):
    """
    Solver for problem 1, compute derivatives and errors
    :param f: function
    :param df: derivative of the function, only used to compute the error
    :param x: x value
    :param ds: list of deltas
    :param n: name of the function, only used for printing
    :return: none
    """
    global log
    
    log.append(f'Estimating f(x)={n} at x={x0}:')
    for d in ds:
        # Estimate the 1st derivative at x
        estimate = (8 * (f(x + d) - f(x - d)) - (f(x + 2 * d) - f(x - 2 * d))) / (12 * d)
        error = np.abs(estimate - df(x))

        log.append(f'Error at delta={d}:', error)


if __name__ == '__main__':
    # Init the logger
    log = Log()

    # Let's do x=1 as an example
    x0 = 1

    # f(x)=exp(x)
    func = np.exp
    dfunc = np.exp
    # We expect delta=0.0009 to give the smallest error
    deltas = [0.01, 0.001, 0.0009, 0.0001, 0.00001]
    name = 'exp(x)'
    problem1(func, dfunc, x0, deltas, name)

    # f(x)=exp(0.01x)
    func = lambda x: np.exp(0.01 * x)
    dfunc = lambda x: 0.01 * np.exp(0.01 * x)
    # We expect delta=0.04 to give the smallest error
    deltas = [1, 0.1, 0.04, 0.01, 0.001]
    name = 'exp(0.01x)'
    problem1(func, dfunc, x0, deltas, name)

    log.save('1.txt')
