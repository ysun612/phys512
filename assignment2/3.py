import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.logger import Log


def log2coef(fit='cheb', cache=True, order=8):
    """
    Find the polynomial's coefficient of log2 for x in 0.5 to 1
    :param fit: Chebyshev, Legendre or Taylor
    :param cache: use/save to cache
    :param order: order of polynomial, disable cache if changed to avoid confusion
    :return: polynomial coefficients
    """
    # Use a cache so we bo not need to compute every time
    global log2coef_cache

    # Only enable cache is set cache=True
    if cache:
        # Try return from cache
        try:
            return log2coef_cache[fit]
        # If cache dict does not exist, create it and continue
        except NameError:
            log2coef_cache = {}
        # If dict exists, but specific polynomial is not used before, continue
        except KeyError:
            pass

    # Generate 50 points between 0.5 to 1
    xs = np.linspace(0.5, 1)
    ys = np.log2(xs)

    # Scale the points to -1 to 1 for Chebyshev and Legendre
    xs_scaled = xs * 4 - 3

    if fit == 'cheb':
        # Order 8 is determined to give error less than required 1e-6
        coef = np.polynomial.chebyshev.chebfit(xs_scaled, ys, 40)[:order]
    elif fit == 'leg':
        coef = np.polynomial.legendre.legfit(xs_scaled, ys, 40)[:order]
    elif fit == 'poly':
        coef = np.polynomial.polynomial.polyfit(xs, ys, 40)[:order]

    # Save result to cache is cache=True
    if cache:
        log2coef_cache[fit] = coef

    return coef


def mylog2(x, base=np.e, fit='cheb'):
    """
    Natural logarithm using my fit to log2
    :param x: x value, must larger than zero
    :param base: base of the log, default to natual base
    :param fit: fit method, default to Chebyshev
    :return: logarithm of x
    """
    assert fit in ['cheb', 'leg', 'poly']
    # Determine if x is iterable
    try:
        iter(x)
    # If not iterable, i.e. a number
    except TypeError:
        # We can only handle positive x
        assert x > 0

        # Split x into mantissa and exponent
        mantissa, exponent = np.frexp(x)
        # Rescale the mantissa before feeding into Chebyshev and Legendre
        mantissa_scaled = mantissa * 4 - 3

        # Skip change of base computation if base is in 2
        if base != 2:
            # Split the case into mantissa and exponent
            mantissa_base, exponent_base = np.frexp(base)
            # Rescale the mantissa before feeding into Chebyshev and Legendre
            mantissa_base_scaled = mantissa_base * 4 - 3

        # Compute log2(mantissa) using the fit for log2 function
        if fit == 'cheb':
            log2mantissa = np.polynomial.chebyshev.chebval(
                mantissa_scaled, log2coef(fit=fit))
            if base != 2:
                log2mantissa_base = np.polynomial.chebyshev.chebval(
                    mantissa_base_scaled, log2coef(fit=fit))
        elif fit == 'leg':
            log2mantissa = np.polynomial.legendre.legval(
                mantissa_scaled, log2coef(fit=fit))
            if base != 2:
                log2mantissa_base = np.polynomial.legendre.legval(
                    mantissa_base_scaled, log2coef(fit=fit))
        elif fit == 'poly':
            log2mantissa = np.polynomial.polynomial.polyval(
                mantissa, log2coef(fit=fit))
            if base != 2:
                log2mantissa_base = np.polynomial.polynomial.polyval(
                    mantissa_base, log2coef(fit=fit))

        if base != 2:
            # log_{a*2^b}(c*2^d)=log2(c*2^d)/log2(a*2^b)=(d+log2(c))/(b+log2(a))
            return (log2mantissa + exponent) / (log2mantissa_base + exponent_base)
        else:
            # log2(c*2^d)=(d+log2(c))
            return log2mantissa + exponent
    # If iterable, we iterate over all x and call ndiff for each of them
    else:
        result = []
        for y in x:
            result.append(mylog2(y, base=base, fit=fit))

        # Return a list
        return result


if __name__ == '__main__':
    log = Log()

    # List of orders to plot
    os = range(1, 18)
    # Generate 50 points between 0.5 and 1
    zs = np.linspace(0.5, 1)
    # Our Chebyshev need x scaled to -1 to 1
    zs_scaled = zs * 4 - 3
    # Max errors
    es = []
    # Compute the maximum difference
    for o in os:
        c = log2coef(cache=False, order=o)
        vs = np.polynomial.chebyshev.chebval(zs_scaled, c)
        es.append(np.max(np.abs(vs - np.log2(zs))))

    plt.figure()
    plt.scatter(os, es)
    plt.xlabel('Chebyshev order')
    plt.ylabel('Max error')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('3.pdf')

    # Create a log sequence
    s = np.logspace(0.1, 10)
    # Compute ln using my implementation
    cheb = np.array(mylog2(s))
    leg = np.array(mylog2(s, fit='leg'))
    poly = np.array(mylog2(s, fit='poly'))
    # Compute ln using NumPy
    real = np.log(s)
    # Compute RMS and max error
    log.append('My implementation using Chebyshev has RMS error:', np.std(cheb - real))
    log.append('My implementation using Chebyshev has max error:', np.max(np.abs(cheb - real)))
    log.append('My implementation using Legendre has RMS error:', np.std(leg - real))
    log.append('My implementation using Legendre has max error:', np.max(np.abs(leg - real)))
    log.append('My implementation using Taylor has RMS error:', np.std(poly - real))
    log.append('My implementation using Taylor has max error:', np.max(np.abs(poly - real)))

    log.save('3.txt')
