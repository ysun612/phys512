import numpy as np

from assignment5 import chi2, error_realization, do_plot, do_print

if __name__ == '__main__':
    # Initialize MCMC
    num_step = 2500
    # Parameters from test script
    p0 = np.array([60, 0.02, 0.1, 0.05, 2.00e-9, 1.0])
    chi2_mcmc = np.zeros(num_step)
    p_mcmc = np.zeros((num_step, p0.size))
    chi2_mcmc[0] = chi2(p0)
    p_mcmc[0, :] = p0
    step = 1

    # Resume from previous result
    tmp = np.loadtxt('planck_chain.txt')
    chi2_mcmc[:len(tmp)] = tmp[:, -1]
    p_mcmc[:len(tmp), :] = tmp[:, :-1]
    step = len(tmp)

    # Read covariance matrix from part 2)
    cov_mat = np.loadtxt('planck_fit_cov_mat.txt')

    # MCMC
    counter = 0
    while step < num_step:
        # Generate step size using covariance matrix
        step_size = error_realization(cov_mat)
        # Tweaked scale for best fit
        p_new = p_mcmc[step - 1] + step_size
        chi2_new = chi2(p_new)
        counter += 1

        # MCMC step, we require all parameters to be greater than 0
        if np.random.rand() < np.exp(-0.5 * (chi2_new - chi2_mcmc[step - 1])) and (p_new > 0).all():
            p_mcmc[step] = p_new
            chi2_mcmc[step] = chi2_new
            step += 1

            print(f'On step {step}, params are {p_new}, chi2 is {chi2_new}, accept rate is {step / counter}')
            res_mcmc = np.column_stack([p_mcmc, chi2_mcmc])
            np.savetxt('planck_chain.txt', res_mcmc[:step])

    res_mcmc = np.column_stack([p_mcmc, chi2_mcmc])
    # Save the MCMC chain
    np.savetxt('planck_chain.txt', res_mcmc)

    # Make the plots
    do_plot(p_mcmc, chi2_mcmc, num_step, '1c')

    # Print fit results
    # Take last 2000 converged steps
    p = np.mean(p_mcmc[-2000:, :], axis=0)
    sigmap = np.std(p_mcmc[-2000:, :], axis=0)
    log = do_print(p, sigmap, 'Fit using MCMC gives:', '1c')
    # Compute OmegaLambda
    h = p[0] / 100
    sigmah = sigmap[0] / 100
    omegalambda = 1 - p[1] / h ** 2 - p[2] / h ** 2
    sigmaomegalambda = np.sqrt(
        (sigmap[1] / p[1]) ** 2 +
        (2 * sigmah / h) ** 2 +
        (sigmap[2] / p[2]) ** 2 +
        (2 * sigmah / h) ** 2
    )
    log.append(f'OmegaLambda is {omegalambda} +- {sigmaomegalambda}')
    log.save('1c.txt')
