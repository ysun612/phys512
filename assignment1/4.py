import numpy as np
from scipy.interpolate import interp1d, lagrange

import sys
sys.path.append("../")
from utils.logger import Log


# The following code is taken/adapted from ratfit_exact.py
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

def rat_fit_p(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q
# Copied code end here


def problem4(f, bound, name, num_point=8, n=4, m=5, num_sample=100):
    """
    Function for q4 of assignment 1
    :param f: input function to interpolate
    :param bound: upper bound
    :param name: name of the function
    :param num_point: number of data points generated from the function
    :param n: order of p(x)
    :param m: order of q(x)
    :param num_sample: number of points to test the error
    :return: none
    """
    global log
    log.append(f'Considering for f(x)={name}:')

    # First we sort the input x to make sure it is monotonic increasing
    x = np.sort((np.random.random(num_point) - 0.5) * bound * 2)
    # Compute the corresponding y value
    y = f(x)

    # These are the evenly spaced points used for evaluting error
    xs = np.linspace(x.min(), x.max(), num=num_sample)

    # Polynomial interpolation
    poly = lagrange(x, y)
    ys = poly(xs)
    log.append('Polynomial error:', (ys - f(xs)).std())

    # Cubic spline interpolation
    interp = interp1d(x, y, kind='cubic')
    ys = interp(xs)
    log.append('Spline error:', (ys - f(xs)).std())

    # Rational function interpolation using numpy.linalg.inv
    p, q = rat_fit(x, y, n, m)
    ys = rat_eval(p, q, xs)
    log.append('Rational function error:', (ys - f(xs)).std())
    log.append('p =', p)
    log.append('q =', q)

    # Rational function interpolation using numpy.linalg.pinv
    p, q = rat_fit_p(x, y, n, m)
    ys = rat_eval(p, q, xs)
    log.append('Rational function (pinv) error:', (ys - f(xs)).std())
    log.append('p =', p)
    log.append('q =', q)


if __name__ == '__main__':
    # Init the logger
    log = Log()

    # f(x)=cos(x)
    problem4(np.cos, np.pi / 2, 'cos(x)')
    # f(x)=1/(x^2+1)
    problem4(lambda x: 1 / (x ** 2 + 1), 1, '1/(x^2+1)')

    log.save('4.txt')
