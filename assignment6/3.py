import numpy as np

from assignment6 import convolve


if __name__ == '__main__':
    # Some test array
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2])

    print(convolve(x, y))
    # My convolve should give the same result as NumPy's convolve with also pad the input
    print(np.convolve(x, y))
