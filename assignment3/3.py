import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('dish_zenith.txt', sep=' ', header=None, names=['x', 'y', 'z'])

    # Linear in x^2+y^2
    df['x^2+y^2'] = df['x'] ** 2 + df['y'] ** 2
    # Constant term
    df['1'] = 1

    # Define the matrix A
    a_mat_bonus = np.column_stack([df['x^2+y^2'], df['x'], df['y'], df['1']])

    # Compute the least square fit
    m_mat_bonus = np.linalg.inv(a_mat_bonus.T @ a_mat_bonus) @ a_mat_bonus.T @ df['z']
    # Compute physical variables
    a = m_mat_bonus[0]
    x0 = - m_mat_bonus[1] / 2 / a
    y0 = - m_mat_bonus[2] / 2 / a
    z0 = m_mat_bonus[3] - a * x0 ** 2 - a * y0 ** 2

    # Print results
    log = Log()
    log.append('Fit result:')
    log.append('m =', m_mat_bonus)
    log.append('a =', a)
    log.append('x0 =', x0)
    log.append('y0 =', y0)
    log.append('z0 =', z0)
    log.save('3b.txt')

    # Plot of the data
    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(df['x'], df['y'], df['z'])
    # xs, ys = np.meshgrid(
    #     np.linspace(df['x'].min(), df['x'].max()),
    #     np.linspace(df['y'].min(), df['y'].max())
    # )
    # zs = a * ((xs - x0) ** 2 + (ys - y0) ** 2) + z0
    # ax.plot_surface(xs, ys, zs, color='tab:orange', alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # Compute the fitted value at each (x,y)
    df['z_fit'] = a_mat_bonus @ m_mat_bonus
    # Noise is the standard deviation between data and fit results
    noise = np.std(df['z'] - df['z_fit'])
    # Define the matrix N
    n_mat = np.eye(len(df)) * noise ** 2
    # Compute the covariance matrix of the fit parameters
    cov_mat = np.linalg.inv(a_mat_bonus.T @ np.linalg.inv(n_mat) @ a_mat_bonus)
    # Compute the error in the fitted parameter
    param_error = np.sqrt(np.diag(cov_mat))

    # Compute the focal length
    f = 1 / 4 / a
    # Use 1st order error propagation
    f_error = param_error[0] / 4 / a ** 2

    # Print the results
    log = Log()
    log.append('Noise (mm):', noise)
    log.append('Focal length (mm):', f, f_error)
    log.save('3c.txt')

    # Linear in x^2, y^2, xy
    df['x^2'] = df['x'] ** 2
    df['y^2'] = df['y'] ** 2
    df['xy'] = df['x'] * df['y']

    # Define the matrix A for the bonus question
    a_mat_bonus = np.column_stack([df['x^2'], df['x'], df['y^2'], df['y'], df['xy'], df['1']])

    # Compute the least square fit for the bonus question
    m_mat_bonus = np.linalg.inv(a_mat_bonus.T @ a_mat_bonus) @ a_mat_bonus.T @ df['z']
    # Compute physical variables for the bonus question
    theta_bonus = 0.5 * np.arctan(m_mat_bonus[4] / (m_mat_bonus[0] - m_mat_bonus[2]))
    a_bonus = 0.5 * (m_mat_bonus[0] + m_mat_bonus[2] + m_mat_bonus[4] / np.sin(2 * theta_bonus))
    b_bonus = 0.5 * (m_mat_bonus[0] + m_mat_bonus[2] - m_mat_bonus[4] / np.sin(2 * theta_bonus))

    # Compute the focal length for the bonus question
    f_bonus_a = 1 / 4 / a_bonus
    f_bonus_b = 1 / 4 / b_bonus

    # Print the results for the bonus question
    log = Log()
    log.append('Focal length in x (mm):', f_bonus_a)
    log.append('Focal length in y (mm):', f_bonus_b)
    log.save('3d.txt')
