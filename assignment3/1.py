import numpy as np

import sys
sys.path.append("../")
from utils.logger import Log


def rk4_step(fun, x, y, h, extra=False, k1=None):
    """
    RK4 step
    :param fun: f(x,y)
    :param x: x
    :param y: y
    :param h: step
    :param extra: also return k1 if True
    :param k1: precomputed value of k1
    :return: y+dy
    """
    # Compute k1 is not given
    if k1 is None:
        # One function call to f
        k1 = h * fun(x, y)
    # Another 3 function calls to f
    k2 = h * fun(x + h / 2, y + k1 / 2)
    k3 = h * fun(x + h / 2, y + k2 / 2)
    k4 = h * fun(x + h, y + k3)
    dy = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    if extra:
        return y + dy, k1
    else:
        return y + dy


def rk4_stepd(fun, x, y, h):
    """
    Optimized RK4 step with half step width
    :param fun: f(x,y)
    :param x: x
    :param y: y
    :param h: step
    :return: y+dy
    """
    # First time we call rk4_step, we save k1
    step_h, k1 = rk4_step(fun, x, y, h, extra=True)

    # Reuse the k1 from previous call, need to convert since h is different
    step_h1 = rk4_step(fun, x, y, h / 2, k1=k1 / 2)
    step_h2 = rk4_step(fun, x + h / 2, step_h1, h / 2)

    # Cancel out the next order term
    error = step_h2 - step_h
    return step_h2 + error / 15


if __name__ == '__main__':
    log = Log()

    # Define the ODE
    f = lambda x, y: y / (1 + x ** 2)
    x_start = -20
    x_end = 20
    y_init = 1

    # Solve using regular RK4
    n = 200
    xs = np.linspace(x_start, x_end, num=n + 1)
    ys = np.empty(n + 1)
    ys[0] = y_init
    for i in range(n):
        ys[i + 1] = rk4_step(f, xs[i], ys[i], xs[1] - xs[0])

    # Solve using optimized half-step RK4
    nd = round(n * 4 / 11)
    xsd = np.linspace(x_start, x_end, num=nd + 1)
    ysd = np.empty(nd + 1)
    ysd[0] = y_init
    for i in range(nd):
        ysd[i + 1] = rk4_stepd(f, xsd[i], ysd[i], xsd[1] - xsd[0])

    # Real solution
    c0 = y_init / np.exp(np.arctan(x_start))
    sol = lambda x: c0 * np.exp(np.arctan(x))
    # Compute the standard deviation to the real value
    yt = sol(xs)
    ytd = sol(xsd)
    log.append('Error using rk4_step:', np.std(yt - ys))
    log.append('Error using rk4_stepd:', np.std(ytd - ysd))

    log.save('1.txt')
