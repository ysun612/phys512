import random
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def libc_rand_vec(num, upper):
    """
    Generate random numbers using glibc, adapted from test_broken_libc.py
    :param num: approximate number of randoms
    :param upper: upper bound
    :return: np. array
    """
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    libc_rand = libc.rand
    libc_rand.argtypes = []
    libc_rand.restype = ctypes.c_int

    @njit
    def __helper(nt):
        return np.array([libc_rand() for _ in range(3 * nt)])

    num_tot = num * 10_000
    gen_raw = np.reshape(__helper(num_tot), [num_tot, 3])
    gen = gen_raw[np.max(gen_raw, axis=1) < upper, :]

    return gen


def linear_combination(x, y, s):
    """
    Linear combination of x and y
    :param x: x
    :param y: x
    :param s: slope for x
    :return: linear combination
    """
    return s * x + y


if __name__ == '__main__':
    # Load provides random numbers
    rand = np.loadtxt('rand_points.txt', dtype=np.int32)

    # We determine the appropriate linear combination by iterating over different slopes
    # plt.figure()
    # plt.tight_layout()
    # for slope in np.linspace(-5, 5, num=20):
    #     plt.clf()
    #     plt.plot(linear_combination(rand[:, 0], rand[:, 1], slope), rand[:, 2],
    #              marker='.', linestyle='', markersize=1)
    #     plt.title(f'${slope}x+y$')
    #     plt.pause(0.01)
    #     plt.show()

    # Find a slope that gives clear set of planes
    slope = -2
    plt.figure()
    plt.plot(linear_combination(rand[:, 0], rand[:, 1], slope), rand[:, 2],
             marker='.', linestyle='', markersize=1)
    plt.xlabel(f'${slope}x+y$')
    plt.ylabel('$z$')
    plt.tight_layout()
    plt.savefig('1.pdf')

    # Parameters for random numbers we generate
    n = 30_000
    m = 100_000_000

    # Generate using Python's random
    rand_py = np.reshape(np.array([random.randint(0, m) for _ in range(3 * n)], dtype=np.int32), [n, 3])
    np.savetxt('rand_points_py.txt', rand_py, fmt='%d')

    # Generate using my computer's glibc 2.36
    rand_libc_my = libc_rand_vec(n, m)
    np.savetxt('rand_points_my.txt', rand_py, fmt='%d')
