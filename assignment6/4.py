import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


def sin_ft(kp, k, n):
    """
    Fourier transform of sin
    :param kp: k'
    :param k: k
    :param n: n
    :return: FT(sin)
    """
    return 1 / 2J * (
        (1 - np.exp(-2 * np.pi * 1J * (kp - k))) / (1 - np.exp(-2 * np.pi * 1J * (kp - k) / n)) -
        (1 - np.exp(-2 * np.pi * 1J * (kp + k))) / (1 - np.exp(-2 * np.pi * 1J * (kp + k) / n))
    )


if __name__ == '__main__':
    # Define basic params
    k = 100 * np.e
    n = 1000

    # Create x and y
    x = np.arange(n)
    y = np.sin(2 * np.pi * k * x / n)
    # Find FFT(y)
    y_ft = np.fft.rfft(y)
    y_ft_analytic = sin_ft(x, k, n)[0:n // 2 + 1]

    log = Log()
    log.append(f'Maximum residue is {np.max(np.abs(y_ft - y_ft_analytic))}')
    log.append(f'Average residue is {np.mean(np.abs(y_ft - y_ft_analytic))}')
    log.save('4c.txt')

    # Make the plot
    plt.figure()
    fig, (ax, residuals) = plt.subplots(nrows=2, ncols=1, sharex='all',
                                        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    ax.plot(y_ft, label='NumPy FFT')
    ax.plot(y_ft_analytic, label='Analytic DFT')
    ax.set_ylabel('Intensity')
    ax.legend()
    residuals.plot(y_ft - y_ft_analytic)
    residuals.set_xlabel('Index')
    residuals.set_ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('4c.pdf')

    # Define the window function
    window = 0.5 - 0.5 * np.cos(2 * np.pi * x / n)
    y_window = y * window
    y_window_ft = np.fft.rfft(y_window)

    plt.figure()
    plt.plot(y_window_ft, label='Window')
    plt.plot(y_ft, label='Raw')
    plt.xlabel('Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('4d.pdf')

    window_ft = np.fft.fft(window)
    log = Log()
    log.append(f'First 5 terms of FFT of the windows function is \n{window_ft[:5]}')
    log.append(f'Last 5 terms of FFT of the windows function is \n{window_ft[-5:]}')
    log.save('1e.txt')
