"""Temporal features calculation module.
"""
import os

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from features.tsfel.utils import compute_time, kde, gaussian


def autocorr(signal):
    """Computes autocorrelation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """
    signal = np.array(signal)
    return float(np.correlate(signal, signal))


def calc_centroid(signal, fs):
    """Computes the centroid along the time axis.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which centroid is computed
    fs: int
        Signal sampling frequency

    Returns
    -------
    float
        Temporal centroid

    """

    time = compute_time(signal, fs)

    energy = np.array(signal) ** 2

    t_energy = np.dot(np.array(time), np.array(energy))
    energy_sum = np.sum(energy)

    if energy_sum == 0 or t_energy == 0:
        centroid = 0
    else:
        centroid = t_energy / energy_sum

    return centroid


def negative_turning(signal):
    """Computes number of negative turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which minimum number of negative turning points are counted
    Returns
    -------
    float
        Number of negative turning points

    """
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal+1] > 0))[0]

    return len(negative_turning_pts)


def positive_turning(signal):
    """Computes number of positive turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which  positive turning points are counted

    Returns
    -------
    float
        Number of positive turning points

    """
    diff_sig = np.diff(signal)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal+1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)


def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which mean absolute deviation is computed

   Returns
   -------
   float
       Mean absolute difference result

   """
    return np.mean(np.abs(np.diff(signal)))


def mean_diff(signal):
    """Computes mean of differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which mean of differences is computed

   Returns
   -------
   float
       Mean difference result

   """
    return np.mean(np.diff(signal))


def median_abs_diff(signal):
    """Computes median absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which median absolute difference is computed

   Returns
   -------
   float
       Median absolute difference result

   """
    return np.median(np.abs(np.diff(signal)))


def median_diff(signal):
    """Computes median of differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which median of differences is computed

   Returns
   -------
   float
       Median difference result

   """
    return np.median(np.diff(signal))


def distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hipotenusa between 2 datapoints.

   Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance

    """
    diff_sig = np.diff(signal)
    return np.sum([np.sqrt(1 + diff_sig ** 2)])


def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which sum absolute difference is computed

   Returns
   -------
   float
       Sum absolute difference result

   """
    return np.sum(np.abs(np.diff(signal)))


def zero_cross(signal):
    """Computes Zero-crossing rate of the signal.

    Corresponds to the total number of times that the signal changes from
    positive to negative or vice versa.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the zero-crossing rate are computed

    Returns
    -------
    int
        Number of times that signal value cross the zero axis

    """
    return len(np.where(np.diff(np.sign(signal)))[0])


def total_energy(signal, fs):
    """Computes the total energy of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which total energy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Total energy

    """
    time = compute_time(signal, fs)

    return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])


def slope(signal):
    """Computes the slope of the signal.

    Slope is computed by fitting a linear equation to the observed data.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which linear equation is computed

    Returns
    -------
    float
        Slope

    """
    t = np.linspace(0, len(signal) - 1, len(signal))

    return np.polyfit(t, signal, 1)[0]


def auc(signal, fs):
    """Computes the area under the curve of the signal computed with trapezoid rule.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed
    fs : int
        Sampling Frequency
    Returns
    -------
    float
        The area under the curve value

    """
    t = compute_time(signal, fs)

    return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))


def abs_energy(signal):
    """Computes the absolute energy of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        Absolute energy

    """
    return np.sum(signal ** 2)


def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        peak to peak distance

    """
    return np.abs(np.max(signal) - np.min(signal))


def entropy(signal, prob='standard'):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value

    """

    a_t = 'the probability function must be either `standard`, `kde` or `gauss`'
    assert prob in ['standard', 'kde', 'gaussian'], a_t
    if prob == 'standard':
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == 'kde':
        p = kde(signal)
    else:
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return - np.sum(p * np.log2(p)) / np.log2(len(signal))


def neighbourhood_peaks(signal, n_factor=0.1):
    """Computes the number of peaks from a defined neighbourhood of the signal.

    Reference: Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018). Time Series FeatuRe Extraction on basis
     of Scalable Hypothesis tests (tsfresh -- A Python package). Neurocomputing 307 (2018) 72-77

    Parameters
    ----------
    signal : nd-array
         Input from which the number of neighbourhood peaks is computed
    n_factor :  float
        Number of peak's neighbours to the left and to the right.
        As a factor of the signal size.

    Returns
    -------
    int
        The number of peaks from a defined neighbourhood of the signal
    """
    n = int(n_factor * len(signal))
    signal = np.array(signal)
    subsequence = signal[n:-n]
    # initial iteration
    peaks = ((subsequence > np.roll(signal, 1)[n:-n]) & (subsequence > np.roll(signal, -1)[n:-n]))
    for i in range(2, n + 1):
        peaks &= (subsequence > np.roll(signal, i)[n:-n])
        peaks &= (subsequence > np.roll(signal, -i)[n:-n])
    return np.sum(peaks)


# dictionary gathering references to the temporal feature functions
FEATURE_FUNCTIONS = {
    'Absolute energy': abs_energy,
    'Area under the curve': auc,
    'Autocorrelation': autocorr,
    'Centroid': calc_centroid,
    'Entropy': entropy,
    'Mean absolute diff': mean_abs_diff,
    'Mean diff': mean_diff,
    'Median absolute diff': median_abs_diff,
    'Median diff': median_diff,
    'Negative turning points': negative_turning,
    'Neighbourhood peaks': neighbourhood_peaks,
    'Peak to peak distance': pk_pk_distance,
    'Positive turning points': positive_turning,
    'Signal distance': distance,
    'Slope': slope,
    'Sum absolute diff': sum_abs_diff,
    'Total energy': total_energy,
    'Zero crossing rate': zero_cross
}
