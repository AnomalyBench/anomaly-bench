"""Forecasting visualization module.

Gathers functions for visualizing period forecasts.
"""
import os

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.forecasting.helpers import get_period_sequence_target_pairs
from visualization.periods.array.common import plot_period


def plot_period_forecasts(period, forecaster, period_labels=None, period_info=None, component_id=None, ax=None):
    """Plots the real values along with the forecaster's predictions for `period`.

    Both real values and forecasts are plotted for `component_id` if specified, else the
    metric shown is the average of all components.

    The `n_back` records used by the forecaster to predict the first value are highlighted
    differently as the 'Period Start': there can be no prediction for these first records.

    Args:
        period (ndarray): period to plot real values and forecaster predictions for.
        forecaster (Forecaster): trained forecaster used to perform forecasts.
        period_labels (ndarray): optional multiclass labels for the period's records.
        period_info (list): optional period info of the form `[file_name, trace_type, period_rank]`.
        component_id (int): optional component to project on. Projected on average if unspecified.
        ax (AxesSubplot): optional plt.axis to plot the data on if not in a standalone figure.

    Returns:
        AxesSubplot: the axis the data was plotted on, to enable further usage.
    """
    # extract (sequence, target) pairs from the period and perform forecasts
    X, Y = get_period_sequence_target_pairs(period, forecaster.n_back, forecaster.n_forward)
    preds = forecaster.predict(X)

    # the first extracted sequence corresponds to the period start, without any prediction
    first_sequence = X[0, :, :]
    component_ids = None if component_id is None else [component_id]
    if ax is None:
        # create the standalone figure plotting the first sequence
        ax = plot_period(
            first_sequence, period_info=period_info,
            curve_specs={'color': 'aqua', 'label': 'Period Start'}, component_ids=component_ids
        )
    else:
        # use the axis of an existing figure to plot the first sequence
        ax = plot_period(
            first_sequence, ax=ax, period_info=period_info,
            curve_specs={'color': 'aqua', 'label': 'Period Start'}, component_ids=component_ids
        )

    # prepend targets and predictions to the first sequence by shifting them `n_back` to the right
    prefix = np.empty((forecaster.n_back, Y.shape[1]))
    prefix[:] = None
    Y = np.insert(Y, 0, prefix, axis=0)
    preds = np.insert(preds, 0, prefix, axis=0)
    plot_period(
        Y, ax=ax, period_labels=period_labels,
        curve_specs={'color': 'blue', 'label': 'Ground-Truth'}, component_ids=component_ids
    )
    plot_period(
        preds, ax=ax,
        curve_specs={'color': 'orange', 'label': 'Forecast'}, component_ids=component_ids
    )
    return ax
