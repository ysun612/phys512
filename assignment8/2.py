import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import sys
sys.path.append("../")
from utils.logger import Log


if __name__ == '__main__':
    l = 200
    # Generate x and y coordinates
    xs = np.linspace(- l / 2, l / 2, l + 1)
    ys = np.linspace(- l / 2, l / 2, l + 1)
    # the coordinate for the origin
    o = l // 2
    x, y = np.meshgrid(xs, ys)

    # Generate V(x, y) and rho(x, y) without normalization
    v = np.log(np.sqrt(x ** 2 + y ** 2))
    v[o, o] = (4 * v[o + 1, o]
               - v[o + 2, o]
               - v[o + 1, o + 1]
               - v[o + 1, o - 1])
    rho = v - (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)) / 4

    # First normalize v and rho such that rho(0,0)=1
    normalization = rho[o, o]
    rho /= normalization
    v /= normalization
    # Then shift v such that v(0,0)=1
    # Shift v does not change rho
    v -= v[o, o] - 1

    log = Log()
    log.append('V(0, 0) =', v[o, o])
    log.append('V(1, 0) =', v[o + 1, o])
    log.append('V(2, 0) =', v[o + 2, o])
    log.append('V(5, 0) =', v[o + 5, o])
    log.save('2a.txt')

    # Reduce the size of the system
    l = 100
    # Regenerate x and y coordinates
    xs = np.linspace(- l / 2, l / 2, l + 1)
    ys = np.linspace(- l / 2, l / 2, l + 1)
    # the coordinate for the origin
    o = l // 2
    x, y = np.meshgrid(xs, ys)

    # Define our box
    l_box = 10
    bc = np.zeros((l + 1, l + 1))
    bc[o - l_box:o + l_box, o - l_box:o + l_box] = 1
    mask = (bc == 1)
    rho = np.zeros((l + 1, l + 1))
    # Green function is V(x,y) we find in a)
    green = v.copy()

    # Find Ax in conjugate gradient
    def ax(green, rho, mask):
        # Set rho outside the box to zero
        rho[~mask] = 0
        # Convolution
        res = fftconvolve(green, rho, mode='valid')
        # Set V outside the box to zero
        res[~mask] = 0
        return res

    # Conjugate gradient
    r = bc - ax(green, rho, mask)
    p = r.copy()
    rtr = np.sum(r ** 2)
    counter = 0
    while rtr > 1e-12 and counter < 1000:
        counter += 1
        ap = ax(green, p, mask)
        pap = np.sum(p * ap)
        alpha = rtr / pap
        rho = rho + alpha * p
        r = r - alpha * ap
        rtr_new = np.sum(r ** 2)
        beta = rtr_new / rtr
        p = r + beta * p
        rtr = rtr_new
        # print('current r^2 is ',rtr_new)

    # Make the figure of rho(x,y)
    plt.figure()
    plt.imshow(rho.T)
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout()
    plt.savefig('2b_rho.pdf')

    plt.figure()
    plt.plot(rho[o - l_box, :])
    plt.xlabel('$x$')
    plt.ylabel(r'$\rho$')
    plt.tight_layout()
    plt.savefig('2b_side.pdf')

    # Find terminal V(x,y)
    v = fftconvolve(green, rho, mode='valid')

    # Make the plot of V(x,y)
    plt.figure()
    plt.imshow(v.T)
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout()
    plt.savefig('2c.pdf')

    v_box = v[o - l_box:o + l_box, o - l_box:o + l_box]
    plt.figure()
    plt.imshow(v_box.T)
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout()
    plt.savefig('2c_inside.pdf')

    log = Log()
    log.append('Mean of V(x,y) inside the box is', np.mean(v_box))
    log.append('Stdev of V(x,y) inside the box is', np.std(v_box))
    log.save('2c.txt')

    # Find terminal E field, we want horizontal axis to be x
    ey, ex = np.gradient(v)
    ey, ex = -ey, -ex
    fig, ax = plt.subplots()
    ax.quiver(x, y, ex, ey)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('2c_e.pdf')

