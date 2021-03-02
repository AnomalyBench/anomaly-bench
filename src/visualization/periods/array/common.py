"""Period ndarray visualization module.

Gathers functions for visualizing periods represented as ndarrays.

TODO - factor out the anomalous periods/zones visualizations currently repeated in
TODO - `plot_period` and `plot_period_scores`.
"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from metrics.evaluators import extract_multiclass_ranges_ids
from detection.thresholding.unsupervised_selectors import IQRSelector


def period_wise_figure(plot_func, periods, labels=None, periods_info=None, fig_title=None, **func_args):
    """Plot period-wise items within the same figure using the provided plotting function for each axis.

    Args:
        plot_func (func): plotting function to call for each period in the figure.
        periods (ndarray): period-wise arrays (typically nD components or 1D outlier scores)
        labels (ndarray): optional periods labels.
        periods_info (ndarray): optional periods information.
        fig_title (str): optional figure title.
        **func_args: optional keyword arguments for the plotting function.
    """
    n_periods = periods.shape[0]
    fig, axs = plt.subplots(n_periods, 1, sharex='none')
    fig.set_size_inches(20, 5 * n_periods)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=25, fontweight='bold')
    if n_periods == 1:
        period_labels = None if labels is None else labels[0]
        period_info = None if periods_info is None else periods_info[0]
        plot_func(periods[0], period_labels=period_labels, period_info=period_info, ax=axs, **func_args)
    else:
        for i, ax in enumerate(axs):
            period_labels = None if labels is None else labels[i]
            period_info = None if periods_info is None else periods_info[i]
            plot_func(periods[i], period_labels=period_labels, period_info=period_info, ax=axs[i], **func_args)


def plot_period(period, period_labels=None, period_info=None, component_ids=None, *,
                ax=None, curve_specs=None, cmap='Dark2', share_y=True,
                zones_info=None, sampling_period=None):
    """Plots `period` projected on the specified components or components average.

    If provided, the labels will be used to highlight the anomalous ranges of the period
    if any (with different target zones if `zones_info` and `sampling_period` are given).

    Args:
        period (ndarray): multidimensional period time series to plot.
        period_labels (ndarray): optional multiclass labels for the period's records.
        period_info (list): optional period info of the form `[file_name, trace_type, period_rank]`.
        component_ids (list): optional list of components to project the period on (average if None).
        ax (AxesSubplot): optional plt.axis to plot the period on if not in a standalone figure.
        curve_specs (dict): optional `{'color': ., 'label': .}` for the curve if only one to plot.
        cmap (str): optional color map to use if multiple curves to plot.
        share_y (bool): if multiple curves, whether we want them to share the y-axis or not.
        zones_info (list): optional list of
            `[primary_start, secondary_start, secondary_end, tolerance_end]` used
            for highlighting the 3 anomaly zones of a "fast alert" evaluation.
        sampling_period (str): sampling period used to turn zones timestamps into
            integer indices.

    Returns:
        AxesSubplot: the axis the period was plotted on, to enable further usage.
    """
    if ax is None:
        # standalone figure
        plt.figure(figsize=(20, 5))
        ax = plt.axes()
    if component_ids is None:
        title = 'Projected on Components Average'
        ax.set_ylabel('Components Average', fontsize=15)
    elif len(component_ids) == 1:
        title = f'Projected on Component #{component_ids[0]}'
    else:
        title = f'Projected on Components #{component_ids}'
    if period_info is not None:
        title = f'{period_info[0]} ({period_info[1].upper().replace("_", " ")}) {title}'
    ax.set_title(title, fontsize=20, y=1.08)
    ax.set_xlabel('Time Index', fontsize=15)

    # plot `period` using the specified projection and optional colors/labels
    ax_handles, ax_labels, axes_spec = [], [], dict()
    if component_ids is None or len(component_ids) == 1:
        # one curve
        if curve_specs is None:
            c, la = 'b', 'Components Average' if component_ids is None else f'Component #{component_ids[0]}'
        else:
            c, la = curve_specs['color'], curve_specs['label']
        ax.plot(np.mean(period, axis=1) if component_ids is None else period[:, component_ids[0]], color=c, label=la)
    else:
        # multiple curves
        colors = plt.get_cmap(cmap).colors
        axes_spec = {'Value': component_ids} if share_y else {f'Component #{i}': [i] for i in component_ids}
        ax_nb = 0
        for unit, cols in axes_spec.items():
            # first y-axis
            if ax_nb == 0:
                n_ax = ax
            # other y-axes are shifted to the right
            else:
                n_ax = ax.twinx()
                n_ax.spines['right'].set_position(('axes', 1.0 + (ax_nb - 1) * 0.07))
            for i, c in enumerate(cols):
                n_ax.plot(period[:, c], label=f'Component #{c}', color=colors[ax_nb + i], alpha=0.8)
            n_ax.set_ylabel(unit, fontsize=15)
            xh, xla = n_ax.get_legend_handles_labels()
            # add legend handles and labels corresponding to this axis
            ax_handles += xh
            ax_labels += xla
            ax_nb += 1

    # highlight the anomalous ranges of the period if any
    if period_labels is not None:
        ranges_ids_dict = extract_multiclass_ranges_ids(period_labels)
        # primary, secondary and tolerance zones information if provided
        if zones_info is not None:
            a_t = 'the sampling period must be provided to plot anomaly zones'
            assert sampling_period is not None, a_t
            prim_start, sec_start, sec_end, tol_end = [pd.Timedelta(s) for s in zones_info]
            sampling_period = pd.Timedelta(sampling_period)
            # zone durations in number of records
            z_durations = [
                int((sec_start - prim_start) / sampling_period),
                int((sec_end - sec_start) / sampling_period),
                int((tol_end - sec_end) / sampling_period)
            ]
            # zone start and end offsets, labels and colors
            z_start_offsets = np.concatenate([[0], np.cumsum(z_durations[:-1])])
            z_end_offsets = np.cumsum(z_durations)
            z_labels = [f'{z} Zone' for z in ['Primary', 'Secondary', 'Tolerance']]
            z_colors = ['purple', 'red', 'pink']
        for class_label in ranges_ids_dict:
            anomalous_ranges = ranges_ids_dict[class_label]
            for range_ in anomalous_ranges:
                # end of the range is exclusive
                beg, end = (range_[0], range_[1] - 1)
                if zones_info is None:
                    ax.axvspan(beg, end, color='r', alpha=0.05)
                    for range_idx in [beg, end]:
                        ax.axvline(range_idx, label='Anomaly', color='r')
                else:
                    for start_offset, end_offset, l, c in zip(
                            z_start_offsets, z_end_offsets, z_labels, z_colors
                    ):
                        start_idx, end_idx = beg + start_offset, beg + end_offset
                        ax.axvspan(start_idx, end_idx, color=c, alpha=0.1)
                        for range_idx in [start_idx, end_idx]:
                            ax.axvline(range_idx, label=l, color=c)

    # get all handles and labels for the legend, prepending the ones from other y-axes if any
    handles, labels = ax.get_legend_handles_labels()
    handles = ax_handles + handles
    labels = ax_labels + labels

    # remove duplicate labels and position the legend closer if no axis on the right
    label_dict = dict(zip(labels, handles))
    x_l, y_l = 1.01 if len(axes_spec.keys()) <= 1 else 1.04, 0.8
    ax.legend(label_dict.values(), label_dict.keys(), bbox_to_anchor=(x_l, y_l), loc='center left')
    ax.grid()
    return ax


def plot_period_scores(scores, period_labels=None, period_info=None, *, ax=None,
                       color='darkgreen', zones_info=None, sampling_period=None, zoomed=False):
    """Plots the outlier scores of a period, highlighting anomalous ranges if specified and any.

    If `zones_info` and sampling_period are given, anomalous ranges will be shown
    highlighting the primary, secondary and tolerance zones relevant to the "fast alert"
    evaluation.

    Args:
        scores (ndarray): outlier scores 1d-array to plot.
        period_labels (ndarray): optional multiclass labels for the period's records.
        period_info (list): optional period info of the form `[file_name, trace_type, period_rank]`.
        ax (AxesSubplot): optional plt.axis to plot the period on if not in a standalone figure.
        color (str): optional color for the outlier score curve.
        zones_info (list): optional list of
            `[primary_start, secondary_start, secondary_end, tolerance_end]` used
            for highlighting the 3 anomaly zones of a "fast alert" evaluation.
        sampling_period (str): sampling period used to turn zones timestamps into
            integer indices.
        zoomed (bool): whether or to plot a "zoomed" version of the plot to fit in a paper.

    Returns:
        AxesSubplot: the axis the period was plotted on, to enable further usage.
    """
    # label for the anomalous ranges if any
    anomaly_label = 'Real Anomaly Range'
    if zoomed:
        fontsizes = {'title': 25, 'axes': 25, 'legend': 25, 'ticks': 22}
    else:
        fontsizes = {'title': 20, 'axes': 17, 'legend': 17, 'ticks': 18}

    # standalone figure
    if ax is None:
        plt.figure(figsize=(20, 5))
        ax = plt.axes()
        title = 'Outlier Scores'
        if period_info is not None:
            title = f'{period_info[0]} ({period_info[1].upper().replace("_", " ")}) {title}'
        ax.set_title(title, fontsize=fontsizes['title'], y=1.07)

    # set the trace name and type as title if available
    if period_info is not None:
        title = f'{period_info[0]} ({period_info[1].upper().replace("_", " ")})'
        ax.set_title(title, fontsize=fontsizes['title'], y=1.07)
    ax.set_xlabel('Time Index', fontsize=fontsizes['axes'])
    ax.set_ylabel('Outlier Score', fontsize=fontsizes['axes'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes['ticks'])
    ax.tick_params(axis='both', which='minor', labelsize=fontsizes['ticks'])
    ax.plot(scores, color=color)

    # highlight the anomalous ranges of the period if any
    if period_labels is not None:
        ranges_ids_dict = extract_multiclass_ranges_ids(period_labels)
        # primary, secondary and tolerance zones information if provided
        if zones_info is not None:
            a_t = 'the sampling period must be provided to plot anomaly zones'
            assert sampling_period is not None, a_t
            prim_start, sec_start, sec_end, tol_end = [pd.Timedelta(s) for s in zones_info]
            sampling_period = pd.Timedelta(sampling_period)
            # zone durations in number of records
            z_durations = [
                int((sec_start - prim_start) / sampling_period),
                int((sec_end - sec_start) / sampling_period),
                int((tol_end - sec_end) / sampling_period)
            ]
            # zone start and end offsets, labels and colors
            z_start_offsets = np.concatenate([[0], np.cumsum(z_durations[:-1])])
            z_end_offsets = np.cumsum(z_durations)
            z_labels = [f'{z} Zone' for z in ['Primary', 'Secondary', 'Tolerance']]
            z_colors = ['purple', 'red', 'pink']
        for class_label in ranges_ids_dict:
            anomalous_ranges = ranges_ids_dict[class_label]
            for range_ in anomalous_ranges:
                # end of the range is exclusive
                beg, end = (range_[0], range_[1] - 1)
                if zones_info is None:
                    ax.axvspan(beg, end, color='r', alpha=0.05)
                    for range_idx in [beg, end]:
                        ax.axvline(range_idx, label=anomaly_label, color='r')
                else:
                    for start_offset, end_offset, l, c in zip(
                            z_start_offsets, z_end_offsets, z_labels, z_colors
                    ):
                        start_idx, end_idx = beg + start_offset, beg + end_offset
                        ax.axvspan(start_idx, end_idx, color=c, alpha=0.1)
                        for range_idx in [start_idx, end_idx]:
                            ax.axvline(range_idx, label=l, color=c)

    # remove duplicate labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    if len(label_dict) > 0:
        red_patch = mpatches.Patch(facecolor='#FFF2F2', edgecolor='red', linewidth=1, label=anomaly_label)
        # put the legend inside the plot if zoomed version, else put it outside
        if zoomed:
            ax.legend(
                label_dict.values(), label_dict.keys(), handles=[red_patch],
                loc='upper left', prop={'size': fontsizes['legend']}
            )
        else:
            ax.legend(
                label_dict.values(), label_dict.keys(), bbox_to_anchor=(1.01, 0.8), loc='center left',
                prop={'size': fontsizes['legend']}
            )
    ax.grid()
    return ax


def plot_scores_distributions(periods_scores, periods_labels, restricted_types=None, fig_title=None,
                              type_colors=None, event_types=None, full_output_path=None,
                              cap_values=False, normalize_scores=False):
    """Plots the distributions of the provided `scores` by record types.

    Args:
        periods_scores (ndarray): periods scores of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        periods_labels (ndarray): either binary or multiclass periods labels.
            With the same shape as `periods_scores`.
        restricted_types (list): optional restriction of record types to plot.
            If not None, have to be either `normal` or `anomalous`, or either `normal` or in `event_types`.
        fig_title (str): optional figure title.
        type_colors (dict|None): if multiple anomaly types, colors to use for each type.
            Every type in `event_types` must then be present as a key in `type_colors`. The color for
            `normal` (label 0) is fixed to blue.
        event_types (list|None): names of the event types that we might encounter in the data (if relevant).
        full_output_path (None|str): optional output path to save the figure to (including file name and extension).
        cap_values (bool): Whether to group outlier scores exceeding 1.5 x IQR within a single bin.
        normalize_scores (bool): Whether to normalize outlier scores to the [0, 1] range, to help comparing
            different methods.
    """
    # check optional type restrictions
    if restricted_types is not None:
        if event_types is None:
            a_t = 'restricted types have to be in `{normal, anomalous}`'
            assert len(set(restricted_types) - {'normal', 'anomalous'}) == 0, a_t
        else:
            a_t = 'restricted types have to be either `normal` or in `event_types`'
            assert len(set(restricted_types) - set(['normal'] + event_types)) == 0, a_t

    # records label names
    label_names = {0: 'normal'}
    pos_label_names = {1: 'anomalous'} if \
        event_types is None else {i+1: event_types[i] for i in range(len(event_types))}
    label_names.update(pos_label_names)

    # histograms colors
    if type_colors is None:
        type_colors = {'anomalous': 'orange'}
    else:
        a_t = 'a color must be provided for every type in `event_types`'
        assert event_types is not None and len(set(event_types) - set(type_colors.keys())) == 0, a_t
    colors = dict({'normal': 'blue'}, **type_colors)

    # histogram assignments
    flattened_scores = np.concatenate(periods_scores)
    if cap_values:
        # all values exceeding 1.5 x IQR are grouped in the last bin
        thresholding_args = argparse.Namespace(
            **{'thresholding_factor': 1.5, 'n_iterations': 1, 'removal_factor': 1}
        )
        selector = IQRSelector(thresholding_args, '')
        selector.select_threshold(flattened_scores)
        flattened_scores[flattened_scores > selector.threshold] = selector.threshold
    if normalize_scores:
        flattened_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            flattened_scores.reshape(-1, 1)
        ).flatten()
    flattened_labels = np.concatenate(periods_labels)
    classes = np.unique(flattened_labels)
    # put "normal" class last if it is there so that it is shown above
    if 0 in classes:
        classes = [cl for cl in classes if cl != 0]
        classes.insert(len(classes), 0)
    scores_dict = dict()
    for class_ in classes:
        # only consider scores that are of the restricted types if relevant
        if restricted_types is None or label_names[class_] in restricted_types:
            scores_dict[label_names[class_]] = flattened_scores[flattened_labels == class_]

    # histograms drawing
    n_bins = 30
    bins = np.linspace(0, 1, n_bins+1) if normalize_scores else n_bins
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(fig_title, size=20, y=0.96)
    for k, scores in scores_dict.items():
        hist_label = k.replace('_', ' ').title().replace('Cpu', 'CPU')
        plt.hist(scores, density=True, bins=bins, label=hist_label, color=colors[k],
                 alpha=0.5 if k != 'normal' else 0.7, edgecolor='black', linewidth=1.2)
    # grid and legend
    fontsizes = {'axes': 20, 'legend': 12, 'ticks': 20}
    plt.grid()
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    try:
        for la, new_idx in zip(['Normal'], [0]):
            current_idx = legend_labels.index(la)
            legend_labels[current_idx] = legend_labels[new_idx]
            handle_to_move = handles[current_idx]
            handles[current_idx] = handles[new_idx]
            legend_labels[new_idx] = la
            handles[new_idx] = handle_to_move
    except ValueError:
        pass
    ax.legend(loc='best', prop={'size': fontsizes['legend']}, labels=legend_labels, handles=handles)
    ax.set_xlabel('Outlier Score', fontsize=fontsizes['axes'])
    ax.set_ylabel('Density', fontsize=fontsizes['axes'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes['ticks'])
    ax.tick_params(axis='both', which='minor', labelsize=fontsizes['ticks'])

    # if values were capped and normalized, add a `+` to the last x tick label
    if cap_values and normalize_scores:
        x_ticks_labels = ax.get_xticks().tolist()
        one_index = [i for i, x in enumerate(x_ticks_labels) if f'{x:.1f}' == '1.0'][0]

        def update_x_ticks(x, pos):
            if pos == one_index:
                return f'{x:.1f}+'
            elif pos > one_index:
                return ''
            else:
                return f'{x:.1f}'
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_x_ticks))

    # save the figure as an image if an output path was provided
    if full_output_path is not None:
        print(f'saving scores distributions figure into {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')
