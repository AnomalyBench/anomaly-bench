"""Period DataFrame visualization base class.
"""
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from metrics.evaluators import extract_multiclass_ranges_ids


class DataFrameViewer:
    """Pandas DataFrame visualization class.

    Plots columns of a pd.DataFrame multidimensional time series.

    Args:
        smoothing (int): records of the series are smoothed using a rolling average of `smoothing` seconds.
        cmap_name (str): name of the color map used for plotting the selected columns.
    """
    def __init__(self, smoothing=60, cmap_name='Set2'):
        self.smoothing = smoothing
        self.cmap_name = cmap_name
        # name to characterize the columns visualization
        self.name = 'default'

    @abstractmethod
    def get_period_view(self, period_df):
        """Returns the DataFrame to plot from `period_df`, along with its "axes specification".

        We define an "axes specification" as a dictionary of the form:
        `{shared_axis_unit_1: col_names_1, shared_axis_unit_2: col_names_2, ...}`.
        => Where `shared_axis_unit_{i}` is the label for the {i}th y-axis, shared by
        the `col_names_{i}` columns list.

        Args:
            period_df (pd.DataFrame): period DataFrame to transform for visualization.

        Returns:
            tuple: `(transformed_df, axes_spec)`.
        """

    def plot_periods(self, period_dfs, periods_info=None, fig_title=None, *, zones_info=None):
        """Plots all `period_dfs` within the same figure using the view defined by `get_period_view`.

        Args:
            period_dfs (list): list of period DataFrames to plot.
            periods_info (list): `[file_name, trace_type, period_rank]` corresponding to the periods.
            fig_title (str): optional figure title.
            zones_info (list): optional list of
                `[primary_start, secondary_start, secondary_end, tolerance_end]` used
                for highlighting the 3 anomaly zones of a "fast alert" evaluation.
        """
        n_periods = len(period_dfs)
        fig, axs = plt.subplots(n_periods, 1, sharex='none')
        fig.set_size_inches(20, 4 * n_periods)
        if fig_title is not None:
            fig.suptitle(fig_title, fontsize=25, fontweight='bold')
        if n_periods == 1:
            period_info = None if periods_info is None else periods_info[0]
            self.plot_period(period_dfs[0], period_info, ax=axs)
        else:
            for i, ax in enumerate(axs):
                period_info = None if periods_info is None else periods_info[i]
                self.plot_period(period_dfs[i], period_info, ax=axs[i], zones_info=zones_info)

    def plot_period(self, period_df, period_info=None, *, ax=None, zones_info=None):
        """Plots `period_df` using the view defined by `get_period_view`.

        If the period DataFrame contains an `Anomaly` column, it will be used to highlight
        anomalous ranges.

        Args:
            period_df (pd.DataFrame): the period DataFrame to visualize.
            period_info (list): optional period info of the form `[file_name, trace_type, period_rank]`.
            ax (AxesSubplot): optional plt.axis to plot the period on if not in a standalone figure.
            zones_info (list): optional list of
                `[primary_start, secondary_start, secondary_end, tolerance_end]` used
                for highlighting the 3 anomaly zones of a "fast alert" evaluation.
        Returns:
            AxesSubplot: the axis the period was plotted on, to enable further usage.
        """
        if ax is None:
            # standalone figure
            plt.figure(figsize=(20, 4))
            ax = plt.axes()
        # show the period's total number of records in the title
        tot_records = int((period_df.index[-1] - period_df.index[0]).total_seconds()) + 1
        title = f'{tot_records:,} Records'
        # if period information is available, also include the file name and trace type in the title
        if period_info is not None:
            title = f'{period_info[0]} ({period_info[1].upper().replace("_", " ")}, {title})'
        ax.set_title(title, fontsize=22, y=1.08)

        # plot view of `period_df` with the specified axes sharing and curve colors
        transformed_trace, axes_spec = self.get_period_view(period_df)
        to_plot = transformed_trace.rolling(self.smoothing).mean()
        colors = plt.get_cmap(self.cmap_name).colors
        ax_nb, ax_handles, ax_labels = 0, [], []
        for unit, cols in axes_spec.items():
            # first y-axis
            if ax_nb == 0:
                n_ax = ax
            # other y-axes are shifted to the right
            else:
                n_ax = ax.twinx()
                n_ax.spines['right'].set_position(('axes', 1.0 + (ax_nb - 1) * 0.08))
            for i, c in enumerate(cols):
                to_plot[c].plot(ax=n_ax, c=colors[ax_nb + i], alpha=0.8)
            n_ax.set_ylabel(unit, fontsize=15)
            xh, xla = n_ax.get_legend_handles_labels()
            # add legend handles and labels corresponding to this axis
            ax_handles += xh
            ax_labels += xla
            ax_nb += 1

        # highlight the anomalous ranges of the period if any
        if 'Anomaly' in period_df.columns:
            ranges_ids_dict = extract_multiclass_ranges_ids(period_df['Anomaly'].values)
            # primary, secondary and tolerance zones information if provided
            if zones_info is not None:
                prim_start, sec_start, sec_end, tol_end = [pd.Timedelta(s) for s in zones_info]
                # zone durations in time deltas
                z_durations = [sec_start - prim_start, sec_end - sec_start, tol_end - sec_end]
                # zone start and end offsets, labels and colors
                z_start_offsets = np.concatenate([[pd.Timedelta('0s')], np.cumsum(z_durations[:-1])])
                z_end_offsets = np.cumsum(z_durations)
                z_labels = [f'{z} Zone' for z in ['Primary', 'Secondary', 'Tolerance']]
                z_colors = ['purple', 'red', 'pink']
            for pos_label in ranges_ids_dict:
                anomalous_ranges = ranges_ids_dict[pos_label]
                for range_ in anomalous_ranges:
                    # end of the range is exclusive
                    beg, end = (range_[0], range_[1] - 1)
                    if zones_info is None:
                        ax.axvspan(period_df.index[beg], period_df.index[end], color='r', alpha=0.05)
                        for range_idx in [beg, end]:
                            ax.axvline(period_df.index[range_idx], label='Anomaly', color='r')
                    else:
                        for start_offset, end_offset, l, c in zip(
                                z_start_offsets, z_end_offsets, z_labels, z_colors
                        ):
                            start_t = period_df.index[beg] + start_offset
                            end_t = period_df.index[beg] + end_offset
                            ax.axvspan(start_t, end_t, color=c, alpha=0.1)
                            for t in [start_t, end_t]:
                                ax.axvline(t, label=l, color=c)
        # get all handles and labels for the legend, prepending the ones from other y-axes if any
        handles, labels = ax.get_legend_handles_labels()
        handles = ax_handles + handles
        labels = ax_labels + labels

        # remove duplicate labels and position the legend closer if no axis on the right
        label_dict = dict(zip(labels, handles))
        x_l, y_l = 1.01 if len(axes_spec.keys()) == 1 else 1.04, 0.8
        ax.legend(label_dict.values(), label_dict.keys(), bbox_to_anchor=(x_l, y_l), loc='center left')
        return ax
