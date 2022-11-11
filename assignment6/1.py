import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from assignment6 import shift_array


if __name__ == '__main__':
    # Generate a Gaussian
    y = norm.pdf(np.linspace(-5, 5, num=100))
    # Shift by half the array length
    y_shifted = shift_array(y, len(y) // 2)

    # Make the plot
    plt.figure()
    plt.plot(y, label='Original')
    plt.plot(y_shifted, label='Shifted')
    plt.xlabel('Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1.pdf')
