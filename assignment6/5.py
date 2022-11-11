import numpy as np
import matplotlib.pyplot as plt
import json

from mf_ligo_class import read_template, read_file, smooth_vector

import sys
sys.path.append("../")
from utils.logger import Log


# Change this to the directory containing data files
DATA_DIR = './LOSC_Event_tutorial/'


def planck_taper(n, epsilon=0.1):
    """
    Planck-Taper window function
    :param n: array length
    :param epsilon: parameter epsilon
    :return: array of window function
    """
    x = np.arange(n)
    w = np.empty(n)
    en = int(epsilon * n)
    # Based on the definition from Wikipedia
    w[0] = 0
    w[1:en] = (1 + np.exp(en / x[1:en] - en / (en - x[1:en]))) ** (-1)
    w[en:n // 2] = 1
    w[-n // 2:] = w[:n // 2][::-1]
    return w


if __name__ == '__main__':
    # Read the JSON file containing all important info
    with open(DATA_DIR + 'BBH_events_v3.json') as f:
        events = json.load(f)

    # Initialize logger for each question
    log_5c = Log()
    log_5d = Log()
    log_5e = Log()
    log_5f = Log()

    # # Iterate over all events
    for event in events:
    # for event in ['GW150914']:
        # Read template, we only use plus polarization here
        tp, _ = read_template(DATA_DIR + events[event][f'fn_template'])
        # Dict to store result of the event
        result = {}
        # Iterate over all detectors
        for detector in ['H', 'L']:
            # Read data
            strain, dt, utc = read_file(DATA_DIR + events[event][f'fn_{detector}1'])
            n = len(strain)

            # Define the window function
            window = planck_taper(n)
            # FFT of strain
            strain_ft = np.fft.rfft(window * strain)

            # Estimate the noise
            noise_ft = np.abs(np.fft.fft(window * strain)) ** 2
            # Smooth the noise with medium filter
            noise_ft_smooth = smooth_vector(noise_ft, 10)
            # We want the same length
            noise_ft = noise_ft[:len(strain_ft)]
            noise_ft_smooth = noise_ft_smooth[:len(strain_ft)]
            # Corresponding frequency
            nu = np.arange(len(strain_ft)) / dt / n
            # Setting noise at nu<20 and nu>1500 to inf
            noise_ft_smooth[nu < 20] = np.inf
            noise_ft_smooth[nu > 1500] = np.inf

            # Make the plot of noise
            plt.figure()
            plt.loglog(nu, noise_ft, label='Un-smoothed noise')
            plt.loglog(nu, noise_ft_smooth, label='Smoothed noise')
            plt.xlabel(r'$\nu$')
            plt.ylabel('Signal')
            plt.tight_layout()
            plt.savefig(f'5a_{event}_{detector}.pdf')

            # Whiten data
            strain_ft_white = strain_ft / np.sqrt(noise_ft_smooth)
            tp_ft_white = np.fft.rfft(tp * window) / np.sqrt(noise_ft_smooth)
            tp_white = np.fft.irfft(tp_ft_white)
            # Cross correlation of template and strain
            rhs = np.fft.irfft(strain_ft_white * np.conj(tp_ft_white))

            # Plot whitened data
            plt.figure()
            plt.plot(np.fft.fftshift(rhs))
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.tight_layout()
            plt.savefig(f'5b_{event}_{detector}.pdf')

            # Estimate the noise from matched filter
            noise_estimate = np.mean(np.abs(rhs))
            sig = np.max(np.abs(rhs))
            snr = sig / noise_estimate
            log_5c.append(f'{event} at {detector} detection gives noise {noise_estimate} and SNR {snr}')

            # Estimate the noise from noise model
            noise_analytic = np.sqrt(np.mean(np.abs(np.fft.irfft(tp_ft_white * np.conj(tp_ft_white)))))
            snr_analytic = sig / noise_analytic
            log_5d.append(f'{event} at {detector} noise model gives noise {noise_analytic} and SNR {snr_analytic}')

            # Find frequency
            ps = np.abs(tp_ft_white) ** 2
            nu_detect = np.sum(nu * ps) / np.sum(ps)
            log_5e.append(f'{event} at {detector} have frequency (Hz) {nu_detect}')

            # Save the time of detection
            result[detector] = {}
            result[detector]['nu_detect'] = nu_detect
            result[detector]['snr'] = snr
            result[detector]['snr_analytic'] = snr_analytic
            result[detector]['rhs'] = rhs

        # Compute the total SNR for two detectors
        snr_combined = np.sqrt(result['H']['snr'] ** 2 + result['L']['snr'] ** 2)
        log_5c.append(f'{event} detection gives combined SNR {snr_combined}')
        snr_analytic_combined = np.sqrt(result['H']['snr_analytic'] ** 2 + result['L']['snr_analytic'] ** 2)
        log_5d.append(f'{event} noise model gives combined SNR {snr_analytic_combined}')

        # Compute the time differences of detection between the two detectors
        index_H = np.argmax(np.abs(result['H']['rhs']))
        index_T = np.argmax(np.abs(result['L']['rhs']))
        index_diff = np.abs(index_H - index_T)

        plt.figure()
        plt.plot(np.abs(result['H']['rhs']), label='H')
        plt.plot(np.abs(result['L']['rhs']), label='L')
        plt.xlim((index_H - 100, index_H + 100))
        plt.xlabel('Index')
        plt.ylabel('Signal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'5f_{event}.pdf')

        log_5f.append(f'{event} detection index difference is {index_diff}')
        log_5f.append(f'{event} relative uncertainty in angle localization is {1 / index_diff}')
        log_5f.append(f'{event} absolute uncertainty in angle localization is (rad) {dt * 300}')

    log_5c.save('5c.txt')
    log_5d.save('5d.txt')
    log_5e.save('5e.txt')
    log_5f.save('5f.txt')
