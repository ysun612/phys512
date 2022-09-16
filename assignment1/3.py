import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def lakeshore(v, data):
    """
    Perform interpolation on lakeshore data
    :param v: voltage or list of voltages
    :param data: data loaded by numpy.loadtxt()
    :return: tuple of interpolated value and estimated uncertainty or list of tuples
    """
    # Store data in a DataFrame
    df = pd.DataFrame(data, columns=['t', 'v', 'dt'])

    # Method of interpolation
    method = 'cubic'
    # Used for estimating uncertainty of the interpolation
    # The ratio of data that are sampled
    sample_ratio = 0.5
    # The number of samples
    sample_num = 100

    # Determine if v is iterable
    try:
        iter(v)
    # If not iterable, i.e. a number
    except TypeError:
        # Perform interpolation
        interp = interp1d(df['v'], df['t'], kind=method)

        # Array to store interpolation of samples
        sampled = np.empty(sample_num)
        for i in range(sample_num):
            # Sample the data set
            df_sample = df.sample(frac=sample_ratio)
            # Perform interpolation on the sampled data set
            # Allow extrapolate here bcause v may lie outside the range of the sample
            sampled[i] = interp1d(df_sample['v'], df_sample['t'], kind=method,
                                  fill_value="extrapolate")(v)
        # Compute the standard deviation of the samples
        error = sampled.std()

        return float(interp(v)), error
    # If iterable, we iterate over all v and call lakeshore for each of them
    else:
        result = []
        for w in v:
            result.append(lakeshore(w, data))

        # Return a list
        return result


if __name__ == '__main__':
    # Some testing
    d = np.loadtxt('lakeshore.txt')
    print(lakeshore([0.1, 0.12, 0.19], d))

    plt.figure()
    plt.scatter(d[:, 1], d[:, 0])
    xs = np.linspace(d[:, 1].min(), d[:, 1].max(), num=100)
    ys = np.array([i[0] for i in lakeshore(xs, d)])
    dys = np.array([i[1] for i in lakeshore(xs, d)])
    plt.fill_between(xs, ys - dys, ys + dys, color='tab:orange', alpha=0.2)
    plt.plot(xs, ys, color='tab:orange')
    plt.show()
