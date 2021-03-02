"""Features options module.

Each option corresponds to a set of features to compute and associated
parameters.
"""
import importlib
import functools

import scipy.signal
import numpy as np

# list of possible window features and associated parameters
FEATURES_OPTIONS = [
    # option 0 - use all features without providing parameters (use defaults)
    {
        'statistical': [
            ['ECDF'],
            ['ECDF Percentile'],
            ['ECDF Percentile Count'],
            ['ECDF Slope'],
            ['Histogram'],
            ['Interquartile range'],
            ['Kurtosis'],
            ['Max'],
            ['Mean'],
            ['Mean absolute deviation'],
            ['Median'],
            ['Median absolute deviation'],
            ['Min'],
            ['Root mean square'],
            ['Skewness'],
            ['Standard deviation'],
            ['Variance']
        ],
        'temporal': [
            ['Absolute energy'],
            ['Area under the curve'],
            ['Autocorrelation'],
            ['Centroid'],
            ['Entropy'],
            ['Mean absolute diff'],
            ['Mean diff'],
            ['Median absolute diff'],
            ['Median diff'],
            ['Negative turning points'],
            ['Neighbourhood peaks'],
            ['Peak to peak distance'],
            ['Positive turning points'],
            ['Signal distance'],
            ['Slope'],
            ['Sum absolute diff'],
            ['Total energy'],
            ['Zero crossing rate']
        ],
        'spectral': [
            ['FFT mean coefficient'],
            ['Fundamental frequency'],
            ['Human range energy'],
            ['LPCC'],
            ['MFCC'],
            ['Max power spectrum'],
            ['Maximum frequency'],
            ['Median frequency'],
            ['Power bandwidth'],
            ['Spectral centroid'],
            ['Spectral decrease'],
            ['Spectral distance'],
            ['Spectral entropy'],
            ['Spectral kurtosis'],
            ['Spectral positive turning points'],
            ['Spectral roll-off'],
            ['Spectral roll-on'],
            ['Spectral skewness'],
            ['Spectral slope'],
            ['Spectral spread'],
            ['Spectral variation'],
            ['Wavelet absolute mean'],
            ['Wavelet energy'],
            ['Wavelet entropy'],
            ['Wavelet standard deviation'],
            ['Wavelet variance']
        ]
    },
    # option 1 - use all features explicitly providing all available parameters
    {
        'statistical': [
            ['ECDF', {'d': 10}],
            ['ECDF Percentile', {'percentile': None}],
            ['ECDF Percentile Count', {'percentile': None}],
            ['ECDF Slope', {'p_end': 0.75, 'p_init': 0.5}],
            ['Histogram', {'nbins': 10, 'r': 1}],
            ['Interquartile range'],
            ['Kurtosis'],
            ['Max'],
            ['Mean'],
            ['Mean absolute deviation'],
            ['Median'],
            ['Median absolute deviation'],
            ['Min'],
            ['Root mean square'],
            ['Skewness'],
            ['Standard deviation'],
            ['Variance']
        ],
        'temporal': [
            ['Absolute energy'],
            ['Area under the curve', {'fs': 100}],
            ['Autocorrelation'],
            ['Centroid', {'fs': 100}],
            ['Entropy', {'prob': 'standard'}],
            ['Mean absolute diff'],
            ['Mean diff'],
            ['Median absolute diff'],
            ['Median diff'],
            ['Negative turning points'],
            ['Neighbourhood peaks', {'n_factor': 0.2}],
            ['Peak to peak distance'],
            ['Positive turning points'],
            ['Signal distance'],
            ['Slope'],
            ['Sum absolute diff'],
            ['Total energy', {'fs': 100}],
            ['Zero crossing rate']
        ],
        'spectral': [
            ['FFT mean coefficient', {'fs': 100, 'nfreq': 256}],
            ['Fundamental frequency', {'fs': 100}],
            ['Human range energy', {'fs': 100}],
            ['LPCC', {'n_coeff': 12}],
            ['MFCC', {
                'fs': 100, 'cep_lifter': 22,
                'nfft': 512, 'nfilt': 40,
                'num_ceps': 12, 'pre_emphasis': 0.97
            }],
            ['Max power spectrum', {'fs': 100}],
            ['Maximum frequency', {'fs': 100}],
            ['Median frequency', {'fs': 100}],
            ['Power bandwidth', {'fs': 100}],
            ['Spectral centroid', {'fs': 100}],
            ['Spectral decrease', {'fs': 100}],
            ['Spectral distance', {'fs': 100}],
            ['Spectral entropy', {'fs': 100}],
            ['Spectral kurtosis', {'fs': 100}],
            ['Spectral positive turning points', {'fs': 100}],
            ['Spectral roll-off', {'fs': 100}],
            ['Spectral roll-on', {'fs': 100}],
            ['Spectral skewness', {'fs': 100}],
            ['Spectral slope', {'fs': 100}],
            ['Spectral spread', {'fs': 100}],
            ['Spectral variation', {'fs': 100}],
            ['Wavelet absolute mean', {
                'function': scipy.signal.ricker, 'widths': np.arange(1, 10)
            }],
            ['Wavelet energy', {
                'function': scipy.signal.ricker, 'widths': np.arange(1, 10)
            }],
            ['Wavelet entropy', {
                'function': scipy.signal.ricker, 'widths': np.arange(1, 10)
            }],
            ['Wavelet standard deviation', {
                'function': scipy.signal.ricker, 'widths': np.arange(1, 10)
            }],
            ['Wavelet variance', {
                'function': scipy.signal.ricker, 'widths': np.arange(1, 10)
            }]
        ]
    },
    # option 2 - use only some statistical features
    {
        'statistical': [
            # ['ECDF', {'d': 10}],
            # ['ECDF Percentile', {'percentile': None}],
            # ['ECDF Percentile Count', {'percentile': None}],
            # ['ECDF Slope', {'p_end': 0.75, 'p_init': 0.5}],
            # ['Histogram', {'nbins': 10, 'r': 1}],
            ['Interquartile range'],
            ['Kurtosis'],
            ['Max'],
            ['Mean'],
            ['Mean absolute deviation'],
            ['Median'],
            ['Median absolute deviation'],
            ['Min'],
            ['Root mean square'],
            ['Skewness'],
            ['Standard deviation'],
            ['Variance']
        ]
    },
    # option 3 - use custom sets from all feature domains
    {
        'statistical': [
            ['Max'], ['Min'], ['Median'], ['Interquartile range'],
            ['Mean'], ['Standard deviation'], ['Variance'],
            ['Skewness'], ['Kurtosis'],
            ['Median absolute deviation'], ['Mean absolute deviation'],
            ['Root mean square']
        ],
        'temporal': [
            ['Median diff'], ['Mean diff'],
            ['Sum absolute diff'], ['Median absolute diff'], ['Mean absolute diff'],
            ['Negative turning points'], ['Positive turning points'],
            ['Slope'], ['Zero crossing rate'], ['Peak to peak distance'],
            ['Area under the curve', {'fs': 1}], ['Centroid', {'fs': 1}],
            ['Absolute energy'], ['Autocorrelation'],
            ['Entropy', {'prob': 'standard'}], ['Signal distance'],
            ['Neighbourhood peaks', {'n_factor': 0.1}]
        ],
        'spectral': [
            ['Fundamental frequency', {'fs': 1}],
            ['Maximum frequency', {'fs': 1}], ['Median frequency', {'fs': 1}],
            ['Max power spectrum', {'fs': 1}], ['Power bandwidth', {'fs': 1}],
            ['Spectral roll-on', {'fs': 1}], ['Spectral roll-off', {'fs': 1}],
            ['Human range energy', {'fs': 1}],
            ['Spectral spread', {'fs': 1}], ['Spectral variation', {'fs': 1}],
            ['Wavelet entropy', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10)}],
            ['Wavelet energy', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10), 'mean': True}],
            ['Wavelet absolute mean', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10), 'mean': True}],
            ['Wavelet standard deviation', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10), 'mean': True}],
            ['Wavelet variance', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10), 'mean': True}]
        ]
    },
    # option 4 - try to limit the number of features
    {
        'statistical': [
            ['Max'], ['Min'], ['Median'], ['Interquartile range'],
            ['Mean'], ['Standard deviation'],
        ],
        'temporal': [
            ['Mean diff'],
            ['Mean absolute diff'],
            ['Negative turning points'], ['Positive turning points'],
            ['Slope'], ['Zero crossing rate'], ['Peak to peak distance'],
            ['Autocorrelation'],
            ['Neighbourhood peaks', {'n_factor': 0.1}]
        ],
        'spectral': [
            ['Fundamental frequency', {'fs': 1}],
            ['Maximum frequency', {'fs': 1}],
            ['Max power spectrum', {'fs': 1}],
            ['Wavelet entropy', {'function': scipy.signal.ricker, 'widths': np.arange(1, 10)}],
        ]
    }
]


def get_feature_functions(option_idx):
    """Returns the feature functions and parameters corresponding to `option_idx`.

    Args:
        option_idx (int): option index in `FEATURES_OPTIONS`.

    Returns:
        list: `functools.partial` objects providing the functions to apply with
            their parameters set as defined in the option.
    """
    # option dictionary
    features_option = FEATURES_OPTIONS[option_idx]
    feature_functions = []
    for domain in features_option:
        # get name-function mappings for the domain features
        functions_dict = importlib.import_module(f'features.tsfel.{domain}').FEATURE_FUNCTIONS
        for feature_info in features_option[domain]:
            # function parameters as keyword arguments
            params = dict()
            if len(feature_info) == 2:
                params = feature_info[1]
            feature_functions.append(
                functools.partial(functions_dict[feature_info[0]], **params)
            )
    return feature_functions
