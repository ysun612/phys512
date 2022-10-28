import numpy as np

from assignment5 import df, get_spectrum, grad_numeric, chi2, do_print

import sys
sys.path.append("../")
from utils.logger import Log


def update_lamda(lamda, success):
    """
    Update lambda for LM, adapted from lm_class.py
    :param lamda: lambda
    :param success: whether success
    :return: new lambda
    """
    if success:
        lamda = lamda / 1.5
        if lamda < 0.5:
            lamda = 0
    else:
        if lamda == 0:
            lamda = 1
        else:
            lamda = lamda * 1.5 ** 2
    return lamda


if __name__ == '__main__':
    # # Compute chi2 for part a)
    # log = Log()
    # pars = np.array([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
    # spec = get_spectrum(pars)[:len(df)]
    # log.append('Params in the test script gives chi2', chi2(pars))
    # pars = np.array([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
    # spec = get_spectrum(pars)[:len(df)]
    # log.append('Params closer to accepted value gives chi2', chi2(pars))
    # log.append('Total degrees of freedom is', len(df) - len(pars))
    # log.save('1a.txt')
    #
    # # A LM method with initial value from the test script
    # p = np.array([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
    # l = 1
    # # Consider the uncertainty, since it is significant
    # n_mat_inv = np.diag(1 / df['ddl'] ** 2)
    # d = get_spectrum(p)[:len(df)]
    # grad = grad_numeric(lambda t, p: get_spectrum(p)[:len(df)], df['l'], p)
    # lhs = grad.T @ n_mat_inv @ grad
    # rhs = grad.T @ n_mat_inv @ (df['dl'] - d)
    # for i in range(10):
    #     dp = np.linalg.inv(lhs + l * np.diag(np.diag(lhs))) @ rhs
    #     chi2_new = chi2(p + dp)
    #
    #     if chi2_new < chi2(p):
    #         if l == 0 and np.abs(chi2_new - chi2(p)) < 10:
    #             break
    #         p = p + dp
    #         d = get_spectrum(p)[:len(df)]
    #         grad = grad_numeric(lambda t, p: get_spectrum(p)[:len(df)], df['l'], p)
    #         lhs = grad.T @ n_mat_inv @ grad
    #         rhs = grad.T @ n_mat_inv @ (df['dl'] - d)
    #         l = update_lamda(l, True)
    #     else:
    #         l = update_lamda(l, False)
    #
    #     print(f'On iteration {i}, params are {p}, chi2 is {chi2(p)}, lambda is {l}')
    #
    # sigmap = np.sqrt(np.diag(np.linalg.inv(lhs)))
    # res_lm = np.array((p, sigmap))
    # cov_mat = np.linalg.inv(lhs)
    # # Save the fit params and covariance matrix
    # np.savetxt('planck_fit_params.txt', res_lm)
    # np.savetxt('planck_fit_cov_mat.txt', cov_mat)
    #
    # # Read from generated file
    # # tmp = np.loadtxt('planck_fit_params.txt')
    # # p = tmp[0, :]
    # # sigmap = tmp[1, :]
    #
    # # Print fit results
    # do_print(p, sigmap, 'Fit using LM gives:', '1b')

    num_steps = 10
    for j in range(8, num_steps):
        # A LM method with initial value from the test script
        if j == 8:
            p = np.array([1.74344989e+02, 2.81414667e-02, 1.39659406e-01, 1.81138876e-09, 1.38317004e+00])
        omch2 = 0.1 - 0.1 * (j + 1) / num_steps
        l = 1
        # Consider the uncertainty, since it is significant
        n_mat_inv = np.diag(1 / df['ddl'] ** 2)
        d = get_spectrum(p, force_omch2=omch2)[:len(df)]
        grad = grad_numeric(lambda t, p: get_spectrum(p, force_omch2=omch2)[:len(df)], df['l'], p)
        lhs = grad.T @ n_mat_inv @ grad
        rhs = grad.T @ n_mat_inv @ (df['dl'] - d)
        for i in range(10):
            dp = np.linalg.inv(lhs + l * np.diag(np.diag(lhs))) @ rhs
            chi2_new = chi2(p + dp, force_omch2=omch2)

            if chi2_new < chi2(p, force_omch2=omch2):
                if l == 0 and np.abs(chi2_new - chi2(p, force_omch2=omch2)) < 10:
                    break
                p = p + dp
                d = get_spectrum(p, force_omch2=omch2)[:len(df)]
                grad = grad_numeric(lambda t, p: get_spectrum(p, force_omch2=omch2)[:len(df)], df['l'], p)
                lhs = grad.T @ n_mat_inv @ grad
                rhs = grad.T @ n_mat_inv @ (df['dl'] - d)
                l = update_lamda(l, True)
            else:
                l = update_lamda(l, False)

            print(f'On iteration {i} of step {j}, params are {p}, chi2 is {chi2(p, force_omch2=omch2)}, lambda is {l}')

    # sigmap = np.sqrt(np.diag(np.linalg.inv(lhs)))
    # res_lm = np.array((p, sigmap))
    # cov_mat = np.linalg.inv(lhs)
