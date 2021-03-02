"""OTN-specific period arrays visualization module.
"""
import os

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from visualization.helpers.otn import get_service_locations
from visualization.periods.array.common import period_wise_figure


def plot_service_figure(plot_func, service_no, datasets, labels, datasets_info=None,
                        **func_args):
    """Plots a figure for visualizing all the periods of the provided service number.

    In the title will be shown the datasets in which the periods belong to.

    Args:
        plot_func (func): plotting function to call for each period in the figure.
        service_no (int): service number (i.e. name) to show (the same as its file name).
        datasets (dict): processed datasets of the form `{set_name: set_array}`;
            with `set_array` an ndarray of shape `(n_periods, period_size, n_features)`.
        labels (dict): labels for each dataset of the form `{set_name, set_labels}`;
            with `set_labels` an ndarray of shape `(n_periods, period_size,)`.
        datasets_info (dict): datasets information of the form {`set_name`: `periods_info`};
            with `periods_info` a list of the form `[file_name, trace_type, period_rank]`
            for each period of the set.
        **func_args: optional keyword arguments for the plotting function.
    """
    # chronologically ranked list of periods `(set_name, set_position)` for the service
    service_locations = get_service_locations(service_no, datasets_info)
    periods, periods_info, periods_labels, service_sets = [], [], [], []
    for (period_set, period_idx) in service_locations:
        periods.append(datasets[period_set][period_idx])
        periods_labels.append(labels[period_set][period_idx])
        periods_info.append(datasets_info[period_set][period_idx])
        service_sets.append(period_set.capitalize())
    periods, periods_labels = np.array(periods, dtype=object), np.array(periods_labels, dtype=object)
    period_wise_figure(
        plot_func, periods, periods_labels, periods_info,
        fig_title=f'Service #{service_no} ({", ".join(service_sets)})', **func_args
    )
