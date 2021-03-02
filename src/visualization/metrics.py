"""Metrics visualization module.
"""
import os

import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def plot_pr_curves(recalls, precisions, ts, *,
                   fig_title=None, colors=None, full_output_path=None):
    """Plots the Precision-Recall curve(s) corresponding to the provided `recalls` and `precisions`.

    If `recalls` is a dictionary, a curve will be plotted for each key, optionally using the colors
    specified by `colors` (if those 2 dictionaries are specified, their keys must hence match).
    The keys will also be used for the legend.

    If `recalls` is an ndarray (like `precisions`), a single curve will be plotted,
    optionally using the single color specified by `colors`.

    Args:
        recalls (ndarray|dict): recall(s) values for each threshold.
        precisions (ndarray): precision values for each threshold.
        ts (ndarray): threshold value for each (recall, precision) pair.
        fig_title (str|None): optional figure title.
        colors (dict|str|None): optional curve color(s).
        full_output_path (str|None): path to save the figure to if specified (with file name and extension).
    """
    # create new figure, set title, labels and limits
    fig, ax = plt.subplots()
    if fig_title is None:
        fig_title = 'Precision-Recall Curve'
    fig.suptitle(fig_title, size=14, y=0.96)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    # get minimum and maximum threshold values to highlight (second max if max is inf)
    min_ts, max_ts = ts[0], (ts[-1] if ts[-1] != np.inf else ts[-2])

    # if `recalls` is not a dict, plot a single PR curve
    if type(recalls) != dict:
        assert colors is None or type(colors) != dict, 'color must be passed as a single value if one curve'
        # highlight the minimum and maximum threshold values to show the "direction" of the curve
        ax.scatter([recalls[0], recalls[-1]], [precisions[0], precisions[-1]], color='r', s=75, marker='.')
        for index, text in zip([0, -1], [f'Threshold = {min_ts:.2f}', f'Threshold = {max_ts:.2f}']):
            ax.annotate(text, (recalls[index], precisions[index]), ha='center')
        # plot the PR curve
        c = colors if colors is not None else 'blue'
        ax.plot(recalls, precisions, color=c, label=f'(AUC = {auc(recalls, precisions):.2f})')
    # if `recalls` is a dict, plot one PR curve per key
    else:
        assert colors is None or type(colors) == dict, 'colors must be passed as a dict if multiple curves'
        if 'global' in recalls:
            # only show minimum and maximum thresholds for the "global" curve if it exists
            ax.scatter(
                [recalls['global'][0], recalls['global'][-1]], [precisions[0], precisions[-1]],
                color='r', s=75, marker='.'
            )
            for index, text in zip([0, -1], [f'Threshold = {min_ts:.2f}', f'Threshold = {max_ts:.2f}']):
                ax.annotate(
                    text, (recalls['global'][index], precisions[index]),
                    ha='center', bbox=dict(boxstyle='round', fc='w')
                )
        # plot the PR curves
        for k in recalls:
            c = colors[k] if colors is not None else None
            ax.plot(
                recalls[k], precisions, color=c,
                label=f'{k.replace("_", " ").upper()} (AUC = {auc(recalls[k], precisions):.2f})'
            )
    ax.legend(loc='best')
    ax.grid()

    # save the figure as an image if specified
    if full_output_path is not None:
        print(f'saving PR curve figure into {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')


def plot_evaluation_curve(x, y, ts, curve_type, *,
                          ax=None, fig_title=None, color='orange', label_prefix='', full_output_path=None):
    """Plots the `curve_type` curve corresponding to the provided `x` and `y` rates.

    For the ROC curve, x and y should be the false positive and true positive rates, respectively.
    For the Precision-Recall curve, x and y should be the Recalls and Precisions, respectively.

    Args:
        x (ndarray): x values for each threshold, of shape `(n_elements,)`.
        y (ndarray): y values for each threshold, of shape `(n_elements,)`.
        ts (ndarray): threshold value for each (x, y) pair.
        curve_type (str): must be either `roc` or `pr`, for ROC or Precision-Recall.
        ax (plt.axis|None): plot on the provided axis if not None, else create a new figure.
        fig_title (str|None): custom figure title if not None. Only relevant if `ax` is None.
        color (str): curve color.
        label_prefix (str|None): custom label prefix to describe the curve before its AUC in the legend.
        full_output_path (str|None): if not None, the figure will be saved to this path (w/ file name and extension).

    Returns:
        (plt.axis): the axis the curve was plotted on, to enable plotting additional curves to it.
    """
    assert curve_type in ['roc', 'pr'], 'The curve type to plot must be either `roc` or `pr`'
    # get minimum and maximum threshold values to highlight (second max if max is inf)
    min_ts, max_ts = ts[0], (ts[-1] if ts[-1] != np.inf else ts[-2])
    # create a new figure with the dotted baseline if no axis was passed to plot on
    if ax is None:
        fig, ax = plt.subplots()
        if fig_title is None:
            if curve_type == 'roc':
                fig_title = 'ROC Curve on the Threshold Set'
            else:
                fig_title = 'Precision-Recall Curve on the Threshold Set'
        fig.suptitle(fig_title, size=16, y=1)
        # plot reference line if ROC curve
        if curve_type == 'roc':
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate' if curve_type == 'roc' else 'Recall')
        ax.set_ylabel('True Positive Rate' if curve_type == 'roc' else 'Precision')
        # highlight the minimum and maximum threshold values to show the 'direction' of the curve
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], color='r', s=75, marker='.')
        for index, text in zip([0, -1], [f'Threshold = {min_ts:.2f}', f'Threshold = {max_ts:.2f}']):
            ax.annotate(text, (x[index], y[index]), ha='center', bbox=dict(boxstyle='round', fc='w'))
    # else assume an existing figure
    else:
        fig = ax.gcf()

    # plot the ROC curve using the provided label prefix and color
    ax.plot(x, y, color=color, label=f'{label_prefix} (AUC = {auc(x, y):.2f})')
    ax.legend(loc='best')
    ax.grid()

    # save the curve as an image if specified
    if full_output_path is not None:
        print(f'saving {curve_type.upper()} curve figure into {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')
    return ax
