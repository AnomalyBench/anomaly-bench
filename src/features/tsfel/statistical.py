"""Statistical features calculation module.
"""
import os

import numpy as np
import scipy.stats

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from features.tsfel.utils import calc_ecdf


def hist(signal, nbins=10, r=1):
    """Computes histogram of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from histogram is computed
    nbins : int
        The number of equal-width bins in the given range
    r : float
        The lower(-r) and upper(r) range of the bins

    Returns
    -------
    nd-array
        The values of the histogram

    """
    histsig, bin_edges = np.histogram(signal, bins=nbins, range=[-r, r])  # TODO:subsampling parameter

    return tuple(histsig)


def interq_range(signal):
    """Computes interquartile range of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which interquartile range is computed

    Returns
    -------
    float
        Interquartile range result

    """
    return np.percentile(signal, 75) - np.percentile(signal, 25)


def kurtosis(signal):
    """Computes kurtosis of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which kurtosis is computed

    Returns
    -------
    float
        Kurtosis result

    """
    return scipy.stats.kurtosis(signal)


def skewness(signal):
    """Computes skewness of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which skewness is computed

    Returns
    -------
    int
        Skewness result

    """
    return scipy.stats.skew(signal)


def calc_max(signal):
    """Computes the maximum value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
       Input from which max is computed

    Returns
    -------
    float
        Maximum result

    """
    return np.max(signal)


def calc_min(signal):
    """Computes the minimum value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which min is computed

    Returns
    -------
    float
        Minimum result

    """
    return np.min(signal)


def calc_mean(signal):
    """Computes mean value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean is computed.

    Returns
    -------
    float
        Mean result

    """
    return np.mean(signal)


def calc_median(signal):
    """Computes median of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median is computed

    Returns
    -------
    float
        Median result

    """
    return np.median(signal)


def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)


def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result

    """
    return scipy.stats.median_absolute_deviation(signal, scale=1)


def rms(signal):
    """Computes root mean square of the signal.

    Square root of the arithmetic mean (average) of the squares of the original values.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed

    Returns
    -------
    float
        Root mean square

    """
    return np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


def calc_std(signal):
    """Computes standard deviation (std) of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which std is computed

    Returns
    -------
    float
        Standard deviation result

    """
    return np.std(signal)


def calc_var(signal):
    """Computes variance of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
       Input from which var is computed

    Returns
    -------
    float
        Variance result

    """
    return np.var(signal)


def ecdf(signal, d=10):
    """Computes the values of ECDF (empirical cumulative distribution function) along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    d: integer
        Number of ECDF values to return

    Returns
    -------
    float
        The values of the ECDF along the time axis
    """
    _, y = calc_ecdf(signal)
    if len(signal) <= d:
        return tuple(y)
    else:
        return tuple(y[:d])


def ecdf_slope(signal, p_init=0.5, p_end=0.75):
    """Computes the slope of the ECDF between two percentiles.
    Possibility to return infinity values.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    p_init : float
        Initial percentile
    p_end : float
        End percentile

    Returns
    -------
    float
        The slope of the ECDF between two percentiles
    """
    signal = np.array(signal)
    # check if signal is constant
    if np.sum(np.diff(signal)) == 0:
        return np.inf
    else:
        x_init, x_end = ecdf_percentile(signal, percentile=[p_init, p_end])
        return (p_end - p_init) / (x_end - x_init)


def ecdf_percentile(signal, percentile=None):
    """Computes the percentile value of the ECDF.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    percentile: list
        Percentile value to be computed

    Returns
    -------
    float
        The input value(s) of the ECDF
    """
    signal = np.array(signal)
    if percentile is None:
        percentile = [0.2, 0.8]
    if isinstance(percentile, (float, int)):
        percentile = [percentile]

    # calculate ecdf
    x, y = calc_ecdf(signal)

    if len(percentile) > 1:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return tuple(np.repeat(signal[0], len(percentile)))
        else:
            return tuple([x[y <= p].max() for p in percentile])
    else:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return signal[0]
        else:
            return x[y <= percentile].max()


def ecdf_percentile_count(signal, percentile=None):
    """Computes the cumulative sum of samples that are less than the percentile.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which ECDF is computed
    percentile: list
        Percentile threshold

    Returns
    -------
    float
        The cumulative sum of samples
    """
    signal = np.array(signal)
    if percentile is None:
        percentile = [0.2, 0.8]
    if isinstance(percentile, (float, int)):
        percentile = [percentile]

    # calculate ecdf
    x, y = calc_ecdf(signal)

    if len(percentile) > 1:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return tuple(np.repeat(signal[0], len(percentile)))
        else:
            return tuple([x[y <= p].shape[0] for p in percentile])
    else:
        # check if signal is constant
        if np.sum(np.diff(signal)) == 0:
            return signal[0]
        else:
            return x[y <= percentile].shape[0]


# dictionary gathering references to the statistical feature functions
FEATURE_FUNCTIONS = {
    'Min': calc_min,
    'Max': calc_max,
    'Median': calc_median,
    'Interquartile range': interq_range,
    'Histogram': hist,
    'Mean': calc_mean,
    'Standard deviation': calc_std,
    'Variance': calc_var,
    'Skewness': skewness,
    'Kurtosis': kurtosis,
    'Root mean square': rms,
    'Mean absolute deviation': mean_abs_deviation,
    'Median absolute deviation': median_abs_deviation,
    'ECDF': ecdf,
    'ECDF Percentile': ecdf_percentile,
    'ECDF Percentile Count': ecdf_percentile_count,
    'ECDF Slope': ecdf_slope
}
