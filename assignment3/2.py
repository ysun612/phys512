import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f(t, y):
    """
    ODE
    :param t: time, not used f(t,y)=f(y)
    :param y: vector of elements
    :return: dy/dt
    """
    global halflife
    global mat

    # Try to use ODE mat if already computed
    try:
        return mat @ y
    # Compute the ODE mat
    except NameError:
        # Zero matrix
        mat = np.zeros((len(halflife), len(halflife)))
        # Diagonal terms are -ln(2)/t_1/2, self decay
        mat += np.diag(- np.log(2) / halflife)
        # Subdiagonal terms are ln(2)/t_1/2, decay products
        mat += np.diag(np.log(2) / halflife[:-1], -1)

        return mat @ y


if __name__ == '__main__':
    # Define the units, we can easily change unit by changing the value of ms
    ms = 1
    s = 10 ** 3 * ms
    m = 60 * s
    h = 60 * m
    d = 24 * h
    y = 365 * d
    by = 10 ** 9 * y

    # Define the half life vector, np.inf is used for stable elements
    halflife = np.array([
        4.468 * by,
        24.10 * d,
        6.70 * h,
        245500 * y,
        75380 * y,
        1600 * y,
        3.8235 * d,
        3.10 * m,
        26.8 * m,
        19.9 * m,
        164.3 * ms,
        22.3 * y,
        5.015 * y,
        138.376 * d,
        np.inf
    ])

    # Set up the problem
    # We start at t=1, so we can use log scale
    t_start = 1
    # Init value of y
    y_init = np.zeros(len(halflife))
    y_init[0] = 1
    # Set max time to 2 times the half life of U-238
    t_end = halflife[0] * 2

    # Generate times
    ts = np.logspace(0, np.log10(t_end), num=100000)
    # Solve the ODE using Radau
    sol = solve_ivp(f, [t_start, t_end], y_init, method='Radau', t_eval=ts)

    # Make the figure for Pb-268/U-238
    plt.figure()
    plt.plot(sol.t, sol.y[14, :] / sol.y[0, :])
    plt.xlim(1e18, t_end)
    plt.xscale('log')
    plt.xlabel('Time (ms)')
    plt.ylabel('Ratio of Pb-268 to U-238')
    plt.tight_layout()
    plt.savefig('2b_pb268u238.pdf')

    # Make the figure for Th230/U234
    plt.figure()
    plt.plot(sol.t, sol.y[4, :] / sol.y[3, :])
    plt.xlim(1e13, 1e18)
    plt.xscale('log')
    plt.xlabel('Time (ms)')
    plt.ylabel('Ratio of Th-230 to U-234')
    plt.tight_layout()
    plt.savefig('2b_th230u234.pdf')
