"""Reconstruction visualization module.

Gathers functions for visualizing period or window reconstructions.
"""
import os

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.reconstruction.helpers import get_period_windows
from visualization.periods.array.common import plot_period


def plot_period_avg_reconstruction(period, reconstructor, period_labels=None,
                                   period_info=None, component_id=None, ax=None,
                                   zones_info=None, sampling_period=None):
    """Plots the original values vs. average reconstruction of the provided period.

    We plot a reconstructed record as the average of its reconstructions within the
    windows it belongs to.

    Both real and reconstructed values are plotted for `component_id` if specified, else the
    metric that is shown is the average of all components.

    Note: this function can also be used to plot an original vs. reconstructed window,
    by simply providing the window as the period.

    Args:
        period (ndarray): period to plot the real and average reconstructed values for.
        reconstructor (Reconstructor): trained reconstructor used to reconstruct the data.
        period_labels (ndarray): optional multiclass labels for the period's records.
        period_info (list): optional period info of the form `[file_name, trace_type, period_rank]`.
        component_id (int): optional component to project on. Projected on average if unspecified.
        ax (AxesSubplot): optional plt.axis to plot the data on if not in a standalone figure.
        zones_info (list): optional list of
            `[primary_start, secondary_start, secondary_end, tolerance_end]` used
            for highlighting the 3 anomaly zones of a "fast alert" evaluation.
        sampling_period (str): sampling period used to turn zones timestamps into
            integer indices.

    Returns:
        AxesSubplot: the axis the data was plotted on, to enable further usage.
    """
    # extract and reconstruct all period windows
    X = get_period_windows(period, reconstructor.window_size, 1)
    reconstructed = reconstructor.reconstruct(X)

    # construct lagged periods by varying the offset from 0 to `window_size-1`
    n_windows, window_size, n_features = reconstructed.shape
    lagged_periods, max_p_len = [], -np.inf
    for i, start_idx in enumerate(range(window_size)):
        # lag `i` is the concatenation of non-overlapping windows starting from window `i`
        lagged_period = reconstructed[list(range(start_idx, n_windows, window_size))]
        if lagged_period.shape[0] != 0:
            # reshape to `(n_records, n_features)`
            lagged_period = lagged_period.reshape((lagged_period.shape[0] * lagged_period.shape[1], -1))
            # prepend NaN values to align lagged periods' heads
            nan_head = np.full((i, n_features), np.nan)
            lagged_periods.append(np.concatenate([nan_head, lagged_period], axis=0))
            # update maximum period length
            if lagged_periods[-1].shape[0] > max_p_len:
                max_p_len = lagged_periods[-1].shape[0]
    # append NaN values to align lagged periods' tails based on the maximum length
    for i in range(len(lagged_periods)):
        p_len = lagged_periods[i].shape[0]
        nan_tail = np.full((max_p_len - p_len, n_features), np.nan)
        lagged_periods[i] = np.concatenate([lagged_periods[i], nan_tail], axis=0)
    # lagged periods are aligned with NaN values to match in size
    # => each matching record is the same in the original period but reconstructed
    # => with a different window
    reconstructed_period = np.nanmean(np.array(lagged_periods), axis=0)

    # plot the original vs. reconstructed period
    component_ids = None if component_id is None else [component_id]
    if ax is None:
        # create the standalone figure plotting the original period
        ax = plot_period(
            period, period_info=period_info,
            curve_specs={'color': 'blue', 'label': 'Original'}, component_ids=component_ids
        )
    else:
        # use the axis of an existing figure to plot the original period
        ax = plot_period(
            period, ax=ax, period_info=period_info,
            curve_specs={'color': 'blue', 'label': 'Original'}, component_ids=component_ids
        )
    # period labels are provided at the end to be placed last in the legend
    plot_period(
        reconstructed_period, ax=ax, period_labels=period_labels,
        zones_info=zones_info, sampling_period=sampling_period,
        curve_specs={'color': 'orange', 'label': 'Reconstruction'}, component_ids=component_ids
    )
    return ax
