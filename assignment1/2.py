import numpy as np


def ndiff(fun, x, full=False):
    """
    Numeric 1st order differentiation
    :param fun: function
    :param x: x value of list of x values
    :param full: if True, also print delta and estimated error
    :return: estimate and possibly additional info, type depends on parameters
    """
    # Determine if x is iterable
    try:
        iter(x)
    # If not iterable, i.e. a number
    except TypeError:
        # Float rounding error
        epsilon = 1e-16

        # Guess a sensible delta to compute f'''(x)
        delta = 1e-4
        # Compute f'''(x)
        fppp = ((fun(x + 2 * delta) - fun(x - 2 * delta))
                - 2 * (fun(x + delta) - fun(x - delta))) / (2 * delta ** 3)

        # Estimate the optimal delta
        delta = np.abs(3 * epsilon * fun(x) / fppp) ** (1 / 3)

        # Compute the estimate of 1st derivative
        estimate = (fun(x + delta) - fun(x - delta)) / (2 * delta)
        # Around the upper limit of the estimation's error
        error = fppp * delta ** 3 / 6 + epsilon / delta

        if full:
            return estimate, delta, error
        else:
            return estimate
    # If iterable, we iterate over all x and call ndiff for each of them
    else:
        result = []
        for y in x:
            result.append(ndiff(fun, y, full=full))

        # Return a list
        return result


if __name__ == '__main__':
    # Let's test it with a simple example of f(x)=1/x
    func = lambda x: 1 / x
    dfunc = lambda x: - 1 / x ** 2
    x0 = [-1, 0.01, 0.1, 1, 10, 1000000]

    print(ndiff(func, x0, full=True))
    print(dfunc(np.array(x0)) - np.array(ndiff(func, x0)))
