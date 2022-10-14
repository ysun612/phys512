import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


def lorentzian(t, p):
    """
    Simple Lorentzian function
    :param t: input
    :param p: array of height, center and scale
    :return: function value
    """
    d = p[0] / (1 + (t - p[1]) ** 2 / p[2] ** 2)
    return d


def three_lorentzians(t, p):
    """
    Sum of three Lorentzian functions
    :param t: input
    :param p: array of parameters, see below
    :return: function value
    """
    a, b, c, t0, dt, w = p
    return (
        lorentzian(t, [a, t0, w]) +
        lorentzian(t, [b, t0 - dt, w]) +
        lorentzian(t, [c, t0 + dt, w])
    )


def grad_analytic_lorentzian(t, p):
    """
    Analytic gradient matrix with respect to parameters of simple Lorentzian
    :param t: input
    :param p: see definition of lorentzian(t, p)
    :return: 2D array of gradients
    """
    grad = np.zeros((t.size, p.size))
    grad[:, 0] = 1 / (1 + (t - p[1]) ** 2 / p[2] ** 2)
    grad[:, 1] = 2 * p[0] * p[2] ** 2 * (t - p[1]) / (p[2] ** 2 + (p[1] - t) ** 2) ** 2
    grad[:, 2] = 2 * p[0] * p[2] * (t - p[1]) ** 2 / (p[2] ** 2 + (p[1] - t) ** 2) ** 2
    return grad


def grad_numeric(f, t, p):
    """
    Numeric gradient finder with respective to parameters
    :param f: function
    :param t: input
    :param p: array of parameters
    :return: 2D array of numeric gradients
    """
    grad = np.zeros((t.size, p.size))
    for j in range(p.size):
        # Use a sensible h
        h = 1e-4 * p[j]
        dp = np.zeros(p.size)
        # Change for j-th component
        dp[j] = h
        # See Q1 of assignment 1
        grad[:, j] = (8 * (f(t, p + dp) - f(t, p - dp)) - (f(t, p + 2 * dp) - f(t, p - 2 * dp))) / (12 * h)

    return grad


def chi2(p):
    """
    Helper function to compute chi2 for three Lorentzians
    :param p: array of parameters
    :return: chi2
    """
    global df
    return np.sum((df['d'] - three_lorentzians(df['t'], p)) ** 2)


def error_realization(cov_mat):
    """
    Realization of parameter error
    :param cov_mat: covariance matrix
    :return: array of parameter errors
    """
    # Use Cholesky decomposition method
    l_mat = np.linalg.cholesky(cov_mat)
    return l_mat @ np.random.randn(p.size).T


if __name__ == '__main__':
    # Read in data
    df = pd.DataFrame(
        np.column_stack([np.load('sidebands.npz')[i] for i in ['time', 'signal']]),
        columns=['t', 'd']
    )

    # For (a), we need to guess a set of parameters first
    p = np.array([1, 0.0002, 0.00002])
    # Newton method
    for i in range(10):
        d = lorentzian(df['t'], p)
        # Compute the gradient analytically
        grad = grad_analytic_lorentzian(df['t'], p)
        lhs = grad.T @ grad
        rhs = grad.T @ (df['d'] - d)
        dp = np.linalg.inv(lhs) @ rhs
        p = p + dp

    # Create the plot for (a) to check convergence
    plt.figure()
    plt.scatter(df['t'], df['d'])
    plt.plot(df['t'], lorentzian(df['t'], p), color='tab:orange')
    plt.xlabel('Time $t$')
    plt.ylabel('Signal $d$')
    plt.tight_layout()
    plt.savefig('1a.pdf')

    # Save the output to include in LaTeX
    log = Log()
    log.append('Fit params using Newton are', p)
    log.save('1a.txt')

    log = Log()
    # The noise in data is estimated by stdev(fit values - real values)
    log.append('Estimated noise in data is', np.std(lorentzian(df['t'], p) - df['d']))
    # The uncertainty in parameters is estimated using the covariance matrix
    log.append('Fit param errors using Newton are', np.sqrt(np.diag(np.linalg.inv(lhs))))
    log.save('1b.txt')

    # For (c), we use the same set of parameters as (a)
    p = np.array([1, 0.0002, 0.00002])
    for i in range(10):
        d = lorentzian(df['t'], p)
        # Compute the gradient numerically
        grad = grad_numeric(lorentzian, df['t'], p)
        lhs = grad.T @ grad
        rhs = grad.T @ (df['d'] - d)
        dp = np.linalg.inv(lhs) @ rhs
        p = p + dp

    # Create the plot for (c) to check convergence
    plt.figure()
    plt.scatter(df['t'], df['d'])
    plt.plot(df['t'], lorentzian(df['t'], p), color='tab:orange')
    plt.xlabel('Time $t$')
    plt.ylabel('Signal $d$')
    plt.tight_layout()
    plt.savefig('1c.pdf')

    log = Log()
    log.append('Fit params using Newton and numeric diff are', p)
    log.append('with error', np.sqrt(np.diag(np.linalg.inv(lhs))))
    log.save('1c.txt')

    # For (d), we use the fit results as a part of the initial guess
    p = np.array([1.4, 0.14, 0.14, 0.0002, 0.00002, 0.00002])
    for i in range(10):
        d = three_lorentzians(df['t'], p)
        # Compute the gradient numerically
        grad = grad_numeric(three_lorentzians, df['t'], p)
        lhs = grad.T @ grad
        rhs = grad.T @ (df['d'] - d)
        dp = np.linalg.inv(lhs) @ rhs
        p = p + dp

    log = Log()
    log.append('Fit params using Newton and numeric diff and three Lorentzians are\n', p)
    log.append('with error', np.sqrt(np.diag(np.linalg.inv(lhs))))
    log.save('1d.txt')

    # Plot the residues for (e)
    plt.figure()
    plt.plot(df['t'], df['d'] - three_lorentzians(df['t'], p))
    plt.xlabel('Time $t$')
    plt.ylabel('Residue in signal')
    plt.tight_layout()
    plt.savefig('1e.pdf')

    # Plot the best fit line
    plt.figure()
    plt.plot(df['t'], df['d'], linestyle='', marker='.', label='Data')
    plt.plot(df['t'], three_lorentzians(df['t'], p), label='Best fit')

    chi2_fit = chi2(p)
    num_realization = 100
    chi2_perturbed = np.empty(num_realization)
    cov_mat = np.linalg.inv(lhs)
    # Generate many realizations of errors and compute the chi2
    for i in range(num_realization):
        p_perturbed = p + error_realization(cov_mat)
        chi2_perturbed[i] = chi2(p_perturbed)

        # Plot a few of them
        if i in [0, 1, 2, 3]:
            plt.plot(df['t'], three_lorentzians(df['t'], p_perturbed), linestyle='--', label='Perturbed')

    plt.xlabel('Time $t$')
    plt.ylabel('Signal $d$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1f.pdf')

    log = Log()
    log.append('Average of the perturbed chi2 is', np.mean(chi2_perturbed))
    log.append('Newton best fit chi2 is', chi2_fit)
    log.append('Difference is', np.mean(chi2_perturbed) - chi2_fit)
    log.save('1f.txt')

    # Now we move on to MCMC
    num_step = 10000
    # Same as in (d)
    p0 = np.array([1.4, 0.14, 0.14, 0.0002, 0.00002, 0.00002])
    chi2_mcmc = np.empty(num_step)
    p_mcmc = np.empty((num_step, p0.size))
    chi2_mcmc[0] = chi2(p0)
    p_mcmc[0, :] = p0
    step = 1
    while step < num_step:
        # Generate step size using covariance matrix
        step_size = error_realization(cov_mat)
        # Tweaked scale for best fit
        p_new = p_mcmc[step - 1] + 0.4 * step_size
        chi2_new = chi2(p_new)
        # MCMC, we also impose some constraint
        if (np.random.rand() < np.exp(-0.5 * (chi2_new - chi2_mcmc[step - 1])) and
                p_new[0] > 0 and p_new[1] > 0 and p_new[2] > 0 and p_new[4] > 0 and p_new[5] > 0):
            p_mcmc[step] = p_new
            chi2_mcmc[step] = chi2_new
            step += 1

    # Plot of best fit line
    plt.figure()
    plt.scatter(df['t'], df['d'])
    plt.plot(df['t'], three_lorentzians(df['t'], p_mcmc[-1]), color='tab:orange')
    plt.xlabel('Time $t$')
    plt.ylabel('Signal $d$')
    plt.tight_layout()
    plt.savefig('1g_fit.pdf')

    # Plot of chi2 as a function of steps
    plt.figure()
    plt.plot(np.arange(num_step), chi2_mcmc)
    plt.xlabel('Steps')
    plt.ylabel(r'$\chi^2$')
    plt.tight_layout()
    plt.savefig('1g_chi2.pdf')

    # Plot of each parameter as a function of steps
    fig, axs = plt.subplots(3, 2)
    for i in range(6):
        i1 = i // 2
        i2 = i % 2
        axs[i1, i2].plot(np.arange(num_step), p_mcmc[:, i])
        axs[i1, i2].set_xlabel('Steps')
    plt.tight_layout()
    plt.savefig('1g_params.pdf')

    # Plot of FFT of each parameter as a function of steps
    fig, axs = plt.subplots(3, 2)
    for i in range(6):
        i1 = i // 2
        i2 = i % 2
        axs[i1, i2].loglog(np.abs(np.fft.rfft(p_mcmc[:, i])))
        axs[i1, i2].set_xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('1g_params_fft.pdf')

    log = Log()
    log.append('Fit params using MCMC and three Lorentzians are\n', p_mcmc[-1, :])
    log.append('with error', np.std(p_mcmc, axis=0))
    log.append('MCMC fit chi2 is', chi2_mcmc[-1])
    log.save('1g.txt')

    # For (h), compute the real width
    log = Log()
    # Assuming time in GHz
    w = p_mcmc[-1, 5]
    sigma_w = np.std(p_mcmc, axis=0)[5]
    dt = p_mcmc[-1, 4]
    sigma_dt = np.std(p_mcmc, axis=0)[4]
    w_real = 9 * w / dt
    sigma_w_real = w_real * np.sqrt((sigma_w / w) ** 2 + (sigma_dt / dt) ** 2)
    log.append('Real width is', w_real)
    log.append('with error', sigma_w_real)
    log.save('1h.txt')
