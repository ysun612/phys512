import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from assignment6 import correlation, shift_array

if __name__ == '__main__':
    # Generate a Gaussian
    y = norm.pdf(np.linspace(-5, 5, num=100))

    # Correlation of y with itself
    y_corr = np.real(correlation(y, y))

    plt.figure()
    plt.plot(y, label='Gaussian')
    plt.plot(y_corr, label='Correlation')
    plt.xlabel('Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2a.pdf')

    # Correlation of y with shifted y
    for s in np.arange(10, len(y) // 2, step=10):
        y_shifted = shift_array(y, s)
        y_shifted_corr = np.real(correlation(y, y_shifted))

        plt.figure()
        plt.plot(y, label='Original')
        plt.plot(y_shifted, label='Shifted')
        plt.plot(y_shifted_corr, label='Correlation')
        plt.xlabel('Index')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'2b_{s}.pdf')

