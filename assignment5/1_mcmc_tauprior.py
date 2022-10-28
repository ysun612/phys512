import numpy as np
from scipy.stats import norm

from assignment5 import chi2, error_realization, do_plot, do_print

import sys
sys.path.append("../")
from utils.logger import Log


if __name__ == '__main__':
    # Initialize MCMC
    num_step = 2500
    # Parameters from test script
    p0 = np.array([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
    chi2_mcmc = np.empty(num_step)
    p_mcmc = np.empty((num_step, p0.size))
    chi2_mcmc[0] = chi2(p0)
    p_mcmc[0, :] = p0
    step = 1

    # Resume from previous result
    tmp = np.loadtxt('planck_chain_tauprior.txt')
    chi2_mcmc[:len(tmp)] = tmp[:, -1]
    p_mcmc[:len(tmp), :] = tmp[:, :-1]
    step = len(tmp)

    # Importance sampling using MCMC chain with unconstrained tau
    res_mcmc = np.loadtxt('planck_chain.txt')
    tauprior = 0.0540
    sigma_tauprior = 0.0074
    weight = np.sqrt(sigma_tauprior ** 2 / (res_mcmc[:, 3] - tauprior) ** 2)

    # Print the fit result by importance sampling
    # Weighted mean
    ave = np.average(res_mcmc[-2000:, :-1], axis=0, weights=weight[-2000:])
    # Weighted standard deviation
    diff2 = (res_mcmc[:, :-1] - ave) ** 2
    coef = 1 / (len(weight) - 1) * len(weight) / np.sum(weight)
    std = np.empty(len(ave))
    for i in range(len(ave)):
        std[i] = np.sqrt(np.sum(weight[-2000:] * diff2[-2000:, i]) * coef)
    do_print(ave, std, 'Constraint tau fit using importance sampling gives:', '1d_is')

    # Generate covariance matrix from importance sampling
    cov_mat_tauprior = np.cov(res_mcmc[:, :-1].T, aweights=weight)

    # MCMC
    counter = 0
    while step < num_step:
        # Generate step size using covariance matrix
        step_size = error_realization(cov_mat_tauprior)
        # Tweaked scale for best fit
        p_new = p_mcmc[step - 1] + step_size
        chi2_new = chi2(p_new)
        counter += 1
        # MCMC step, we require all parameters to be greater than 0
        # Constraint for tau is added as a bias in probability
        if np.random.rand() < np.exp(-0.5 * (chi2_new +
                                             (p_new[3] - tauprior) ** 2 / sigma_tauprior ** 2 -
                                             chi2_mcmc[step - 1])) and (p_new > 0).all():
            p_mcmc[step] = p_new
            chi2_mcmc[step] = chi2_new
            step += 1

            print(f'On step {step}, params are {p_new}, chi2 is {chi2_new}, accept rate is {step / counter}')
            res_mcmc_tauprior = np.column_stack([p_mcmc, chi2_mcmc])
            np.savetxt('planck_chain_tauprior.txt', res_mcmc_tauprior[:step])

    res_mcmc_tauprior = np.column_stack([p_mcmc, chi2_mcmc])
    # Save the MCMC chain with constraint tau
    np.savetxt('planck_chain_tauprior.txt', res_mcmc_tauprior)

    # Make the plots
    # do_plot(p_mcmc, chi2_mcmc, num_step, '1d')

    # Print fit results
    p = np.mean(p_mcmc[-2000:, :], axis=0)
    sigmap = np.std(p_mcmc[-2000:, :], axis=0)
    do_print(p, sigmap, 'Constraint tau fit using MCMC gives:', '1d')

    # Print 5 sigma error bar
    log = Log()
    log.append('Constraint tau fit using MCMC have 5sigma error bars:')
    for i, n in zip(range(len(p)), ['H0', 'Omegabh2', 'Omegach2', 'tau', 'As', 'ns']):
        log.append(f'{n}: [{p[i] - 5 * sigmap[i]}, {p[i] + 5 * sigmap[i]}]')
    log.save('1d_5sigma.txt')
