import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def integrate_adaptive(fun, a, b, tol, extra=None):
    """
    My integrator with adaptive step size, adapted from question 2
    :param fun: integrand
    :param a: lower bound
    :param b: upper bound
    :param tol: tolerance
    :param extra: pre-computed data
    :return: integration result
    """
    # 5 evenly spaced points
    x = np.linspace(a, b, num=5)

    # Handle poles
    try:
        # Try-except can only capture error
        with np.errstate(invalid='raise'):
            # If precomputed value is not passed, we computed all five points
            if extra is None:
                y = fun(x)
            # Precomputed value for the first, last and the middle points
            else:
                y = np.array([extra[0], fun(x[1]), extra[1], fun(x[3]), extra[2]])
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


def f(u, z, r=1):
    """
    Integrand for E field of a spherical shell, with no overall constant
    :param u: variable to integrate
    :param z: distance to the center
    :param r: radius of the shell, default to 1
    :return: integrand
    """
    return (z - r * u) / (r ** 2 + z ** 2 - 2 * r * z * u) ** (3 / 2)


if __name__ == '__main__':
    # Generate 1001 points, contains x=r
    zs = np.linspace(0, 5, num=1001)
    # Do the integration using my integrator
    es_my = np.array([integrate_adaptive(lambda u: f(u, z=z), -1, 1, 1e-7) for z in zs])
    # Do the integration using scipy's quad integrator
    es_quad = np.array([quad(lambda u: f(u, z=z), -1, 1)[0] for z in zs])

    # Make figures
    plt.figure()
    plt.plot(zs, es_my, label='My integrator')
    plt.plot(zs, es_quad, linestyle='--', label='Quad')
    plt.xlabel('$z$ ($R$)')
    plt.ylabel(r'$E$ ($\frac{2\pi R^2\sigma}{4\pi\epsilon_0}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1.pdf')
