import numpy as np

import sys
sys.path.append("../")
from utils.logger import Log


# The following code is taken/adapted from integrate_adaptive_class.py
def integrate(fun,a,b,tol):
    global lazycounter

    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    # Call fun 5 times
    lazycounter += 5
    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2)
        int2=integrate(fun,mid,b,tol/2)
        return int1+int2
# Copied code end here


def integrate_adaptive(fun, a, b, tol, extra=None):
    """
    My integrator with adaptive step size, with a function calling counter
    :param fun: integrand
    :param a: lower bound
    :param b: upper bound
    :param tol: tolerance
    :param extra: pre-computed data
    :return: integration result
    """
    # Conuter for counting function calls
    global mycounter

    # 5 evenly spaced points
    x = np.linspace(a, b, num=5)

    # Handle poles
    try:
        # Try-except can only capture error
        with np.errstate(invalid='raise'):
            # If precomputed value is not passed, we computed all five points
            if extra is None:
                y = fun(x)
                # Need to call fun 5 times
                mycounter += 5
            # Precomputed value for the first, last and the middle points
            else:
                y = np.array([extra[0], fun(x[1]), extra[1], fun(x[3]), extra[2]])
                # Need to call fun 2 times only
                mycounter += 2
    except FloatingPointError:
        # If pole exists, we directly return a None
        print('Invalid value encountered during integration, setting result to None.')
        return None

    dx = x[1] - x[0]

    i1 = (y[0] + 4 * y[2] + y[4]) / 3 * (2 * dx)
    i2 = (y[0] + 4 * y[1] + 2 * y[2] + 4 * y[3] + y[4]) / 3 * dx

    # Iterative approach
    err = np.abs(i1 - i2)
    if err < tol:
        return i2
    else:
        mid = (a + b) / 2
        # Pass the computed values to our integrator to reduce function calls on fun
        int1 = integrate_adaptive(fun, a, mid, tol / 2, extra=y[:3])
        int2 = integrate_adaptive(fun, mid, b, tol / 2, extra=y[-3:])
        return int1 + int2


def problem2(fun, a, b, tol, name):
    """
    Helper for question 2
    :param fun: integrand
    :param a: lower bound
    :param b: upper bound
    :param tol: tolerance
    :param name: name of the integrand
    :return: None
    """
    global log, mycounter, lazycounter

    # Initialize counter
    mycounter = 0
    lazycounter = 0

    log.append(f'Integrating f(x)={name} between {a} and {b} with max error {tol}:')
    integrate_adaptive(fun, a, b, tol)
    log.append(f'My integrator called f(x) {mycounter} times')
    integrate(fun, a, b, tol)
    log.append(f'Lazy integrator called f(x) {lazycounter} times')


if __name__ == '__main__':
    log = Log()

    # Let's do the computation for f(x)=cos(x) and f(x)=1/(x^2+1)
    problem2(np.exp, 0, 1, 1e-7, 'exp(x)')
    problem2(np.cos, 0, 1, 1e-7, 'cos(x)')
    problem2(lambda x: 1 / (x ** 2 + 1), 0, 1, 1e-7, '1/(x^2+1)')

    log.save('2.txt')
