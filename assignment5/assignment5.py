import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb

import sys
sys.path.append("../")
from utils.logger import Log


def get_spectrum(pars, lmax=3000, force_omch2=None):
    """
    Get spectrum from CAMB, adapted from test script
    :param pars: parameters
    :param lmax: max l
    :param force_omch2: force omch2 value
    :return: spectrum
    """
    if force_omch2 is None:
        H0, ombh2, omch2, tau, As, ns = pars
    else:
        omch2 = force_omch2
        H0, ombh2, tau, As, ns = pars
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    cmb = powers['total']
    tt = cmb[:, 0]  # you could return the full power spectrum here if you wanted to do say EE
    return tt[2:][:2507]


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


def chi2(p, force_omch2=None):
    """
    Helper function to compute chi2
    :param p: array of parameters
    :param force_omch2: force omch2 value
    :return: chi2
    """
    global df
    return np.sum((df['dl'] - get_spectrum(p, force_omch2=force_omch2)[:len(df)]) ** 2 / df['ddl'] ** 2)


def error_realization(cov_mat):
    """
    Realization of parameter error
    :param cov_mat: covariance matrix
    :return: array of parameter errors
    """
    # Use Cholesky decomposition method
    l_mat = np.linalg.cholesky(cov_mat)
    return l_mat @ np.random.randn(len(cov_mat)).T


def do_plot(p_mcmc, chi2_mcmc, num_step, question):
    """
    Make the plot for MCMC
    :param p_mcmc: parameters
    :param chi2_mcmc: chi2
    :param num_step: number of steps
    :param question: question number
    :return: None
    """
    # Plot of best fit line
    plt.figure()
    plt.scatter(df['l'], df['dl'])
    plt.plot(df['l'], get_spectrum(p_mcmc[-1]), color='tab:orange')
    plt.xlabel('$l$')
    plt.ylabel(r'$\mathrm{d}l$')
    plt.tight_layout()
    plt.savefig(f'{question}_fit.pdf')

    # Plot of chi2 as a function of steps
    plt.figure()
    plt.plot(np.arange(num_step), chi2_mcmc)
    plt.xlabel('Steps')
    plt.ylabel(r'$\chi^2$')
    plt.tight_layout()
    plt.savefig(f'{question}_chi2.pdf')

    # Plot of each parameter as a function of steps
    fig, axs = plt.subplots(3, 2)
    for i, n in zip(range(6), ['H_0', '\\Omega_bh^2', '\\Omega_ch^2', '\\tau', 'A_s', 'n_s']):
        i1 = i // 2
        i2 = i % 2
        axs[i1, i2].plot(np.arange(num_step), p_mcmc[:, i])
        axs[i1, i2].set_xlabel('Steps')
        axs[i1, i2].set_ylabel(f'${n}$')
    plt.tight_layout()
    plt.savefig(f'{question}_params.pdf')

    # Plot of FFT of each parameter as a function of steps
    fig, axs = plt.subplots(3, 2)
    for i, n in zip(range(6), ['H_0', '\\Omega_bh^2', '\\Omega_ch^2', '\\tau', 'A_s', 'n_s']):
        i1 = i // 2
        i2 = i % 2
        axs[i1, i2].loglog(np.abs(np.fft.rfft(p_mcmc[:, i])))
        axs[i1, i2].set_ylabel(f'FFT of ${n}$')
    plt.tight_layout()
    plt.savefig(f'{question}_params_fft.pdf')


def do_print(p, sigmap, title, question):
    """
    Print parameters
    :param p: parameters
    :param sigmap: uncertainties
    :param title: title of the fit
    :param question: question number
    :param save: whether directly save
    :return: Log
    """
    log = Log()
    log.append(title)
    for i, n in zip(range(len(p)), ['H0', 'Omegabh2', 'Omegach2', 'tau', 'As', 'ns']):
        log.append(f'{n} = {p[i]} +- {sigmap[i]}')
    log.append(f'with chi2 {chi2(p)}')
    log.save(f'{question}.txt')

    return log


# Read the data to a DataFrame
df = pd.read_csv('COM_PowerSpect_CMB-TT-full_R3.01.txt',
                 sep='   ', header=None, skiprows=1, names=['l', 'dl', '-ddl', '+ddl'])
# Compute the uncertainty in dl
df['ddl'] = (df['-ddl'] + df['+ddl']) / 2
# A few uncertainty is NA, fill them, so we don't run into error when fitting
df = df.fillna(method='pad')
