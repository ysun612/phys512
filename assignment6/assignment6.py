import numpy as np
from scipy.signal import unit_impulse


def shift_array(arr, shift, real=True):
    """
    Shift a 1D array by shift amount using FFT
    :param arr: input array
    :param shift: amount of shift
    :param real: if input is real
    :return: shifted array
    """
    arr_ft = np.fft.fft(arr)

    # To shift the array, we note (f*delta(y))(x)=f(x-y)
    # Create a delta at shift
    delta = unit_impulse(len(arr), shift)
    delta_ft = np.fft.fft(delta)

    # Multiplication in Fourier space is convolution in real space
    arr_shifted_ft = arr_ft * delta_ft
    arr_shifted = np.fft.ifft(arr_shifted_ft)

    if real:
        return np.real(arr_shifted)
    else:
        return arr_shifted


def correlation(arr1, arr2):
    """
    Correlation of two arrays
    :param arr1: 1st input array
    :param arr2: 2nd input array 2
    :return: array of correlation
    """
    arr1_ft = np.fft.fft(arr1)
    arr2_ft = np.fft.fft(arr2)
    arr = np.fft.ifft(arr1_ft * np.conj(arr2_ft))

    return arr


def convolve(arr1, arr2):
    """
    Convolution with no danger of warping around
    :param arr1: 1st input array
    :param arr2: 2nd input array 2
    :return: array of convolution
    """
    n = len(arr1) + len(arr2) - 1
    # Pad each array to a total length of n with zeros at the end
    arr1_padded = np.pad(arr1, (0, n - len(arr1)))
    arr2_padded = np.pad(arr2, (0, n - len(arr2)))

    return np.fft.ifft(np.fft.fft(arr1_padded) * np.fft.fft(arr2_padded))
