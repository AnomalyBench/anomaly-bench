"""Scoring evaluation module.

Gathers functions for evaluating the ability of the outlier score derivation to separate out
normal from anomalous elements.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from visualization.functions import plot_curves
from visualization.metrics import plot_pr_curves
from visualization.periods.array.common import plot_scores_distributions


def get_column_names(periods_labels, event_types, metric_names, set_name):
    """Returns the column names to use for the evaluation DataFrame.

    Args:
        periods_labels (ndarray): periods labels of shape `(n_periods, period_length,)`.
            Where `period_length` depends on the period.
        event_types (list|None): names of the event types that we might encounter in the data (if relevant).
        metric_names (list): list of metric names to compute.
        set_name (str|None): if not None, the set name will be prepended to the column names.

    Returns:
        list: the column names to use for the evaluation.
    """
    set_prefix = '' if set_name is None else f'{set_name.upper()}_'
    if event_types is None:
        return [f'{set_prefix}{m}' for m in metric_names]
    column_names = []
    # positive classes encountered in the periods of the dataset
    pos_classes = np.delete(np.unique(np.concatenate(periods_labels, axis=0)), 0)
    # event "classes" (`global` + one per type)
    class_names = ['global'] + [event_types[i-1] for i in range(1, len(event_types) + 1) if i in pos_classes]
    for cn in class_names:
        # precision is only considered globally
        class_metric_names = metric_names if cn == 'global' else [m for m in metric_names if m != 'PRECISION']
        column_names += [f'{set_prefix}{cn.upper()}_{m}' for m in class_metric_names]
    return column_names


def get_metrics_row(labels, scores, evaluator, column_names, event_types,
                    granularity, method_path=None, evaluation_string=None, metrics_colors=None,
                    periods_key=None):
    """Returns the metrics row to add to the evaluation DataFrame.

    Args:
        labels (ndarray): periods labels of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        scores (ndarray): periods outlier scores with the same shape as `labels`.
        evaluator (metrics.Evaluator): object defining the binary metrics of interest.
        column_names (list): list of column names corresponding to the metrics to compute.
        event_types (list|None): names of the event types that we might encounter in the data (if relevant).
        granularity (str): evaluation granularity.
            Must be either `global`, for overall, `app`, for app-wise or `trace`, for trace-wise.
        method_path (str|None): scoring method path, to save any extended evaluation to.
        evaluation_string (str|None): formatted evaluation string to compare models under the same requirements.
        metrics_colors (dict|str|None): color to use for the curves if single value, color to use
            for each event type if dict (the keys must then belong to `event_types`).
        periods_key (str): if granularity is not `global`, name to use to identify the periods.
            Has to be of the form `appX` if `app` granularity or `trace_name` if `trace` granularity.

    Returns:
        list: list of metrics to add to the evaluation DataFrame (corresponding to `column_names`).
    """
    assert granularity in ['global', 'app', 'trace']

    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)

    # recover dataset name, prefix and title from the column names
    set_name, set_prefix, set_title = None, f'{column_names[0].split("_")[0]}_', ''
    if set_prefix in ['THRESHOLD_', 'TEST_']:
        set_name = set_prefix[:-1].lower()
        set_title = set_name.capitalize()
    else:
        set_prefix = ''

    # compute the PR curve(s) using the Precision and Recall metrics defined by the evaluator
    f, p, r, pr_ts = evaluator.precision_recall_curves(labels, scores, return_f_scores=True)
    # we do not use average metrics across types here
    for d in f, r:
        d.pop('avg')
    # case of a single event type
    if event_types is None:
        metrics_row.at[0, f'{set_prefix}PR_AUC'] = auc(r['global'], p)
        r, f = r['global'], f['global']
    # case of multiple known event types
    else:
        # `global` and interpretable event types that belong to the keys of recall curves
        class_names = ['global'] + [event_types[i-1] for i in range(1, len(event_types) + 1) if i in r]
        # add the metric and column corresponding to each type
        for cn in class_names:
            if cn == 'global':
                metrics_row.at[0, f'{set_prefix}GLOBAL_PR_AUC'] = auc(r['global'], p)
            else:
                label_key = event_types.index(cn) + 1
                metrics_row.at[0, f'{set_prefix}{cn.upper()}_PR_AUC'] = auc(r[label_key], p)
                # update the label keys to reflect interpretable event types suited for visualizations
                r[cn], f[cn] = r.pop(label_key), f.pop(label_key)

    if granularity in ['global', 'app']:
        # save the distributions of outlier scores grouped by record type
        periods_suffix, fig_title_suffix = '', ''
        if granularity == 'global':
            periods_suffix, fig_title_suffix = '_global', ' Globally'
        elif periods_key is not None:
            periods_suffix, fig_title_suffix = f'_{periods_key}', f' for Application {periods_key[3:]}'
        full_output_path = os.path.join(method_path, f'{set_name}{periods_suffix}_scores_distributions.png')
        a_t = 'the provided colors must be either a dict or `None`'
        assert type(metrics_colors) == dict or metrics_colors is None, a_t
        plot_scores_distributions(
            scores, labels,
            fig_title=f'{set_title} Scores Distributions{fig_title_suffix}',
            type_colors=metrics_colors,
            event_types=event_types,
            full_output_path=full_output_path,
            cap_values=True, normalize_scores=True
        )

    if granularity == 'global':
        # save the full PR curve(s) under the method path
        full_output_path = os.path.join(method_path, f'{evaluation_string}_{set_name}_pr_curve.png')
        plot_pr_curves(
            r, p, pr_ts,
            fig_title=f'Precision-Recall Curve on the {set_title} Set',
            colors=metrics_colors,
            full_output_path=full_output_path
        )
        # save the F-score curve(s) under the method path
        full_output_path = os.path.join(method_path, f'{evaluation_string}_{set_name}_f{evaluator.beta}_curve.png')
        plot_curves(
            pr_ts, f,
            fig_title=f'F{evaluator.beta}-Score on the {set_title} Set',
            xlabel='Threshold',
            ylabel=f'F{evaluator.beta}-Score',
            colors=metrics_colors,
            full_output_path=full_output_path
        )
    return metrics_row.iloc[0].tolist()


def save_scoring_evaluation(data_dict, evaluator, evaluation_string, config_name, spreadsheet_path,
                            *, event_types=None, metrics_colors=None, method_path=None):
    """Adds the scoring evaluation of this configuration to a sorted comparison spreadsheet.

    If a list of types is passed as `event_type`, the evaluation will be both performed "globally"
    (i.e. considering all types together) and type-wise.
    The passed `event_types` can be a subset of the types that are indeed represented in the data,
    i.e. whose `index + 1` appear in the labels.

    If `event_type` is None, the labels will be considered binary. Only the "global" performance
    will be reported and considered as default.

    Note: the term `global` is both used when making no difference between applications or traces
    ("global granularity") and when making no difference between event types ("global type").

    Args:
        data_dict (dict): [threshold and] test periods record-wise outlier scores, labels and information:
            - scores as `(set_name)_(scores): ndarray`.
            - labels as `y_(set_name): ndarray`.
            - info as `(set_name)_info: ndarray`. With each period info of the form `(file_name, event_type, rank)`.
            For the first two array types, shape `(n_periods, period_length)`; `period_length` depending on the period.
             - Shapes `(n_samples, n_back, n_features)` for sequences.
             - Shapes `(n_samples, n_features)` or `(n_samples, n_forward, n_features)` for targets.
        evaluator (metrics.Evaluator): object defining the binary metrics of interest.
        evaluation_string (str): formatted evaluation string to compare models under the same requirements.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.
        event_types (list|None): types of events that we might encounter in the data (if relevant).
            The positive label of an event type is its index in the list +1.
        metrics_colors (dict|str|None): color to use for the curves if single value, color to use
            for each event type if dict (the keys must then belong to `event_types`).
        method_path (str): scoring method path, to save any extended evaluation to.
    """
    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{evaluation_string}_scoring_comparison.csv')

    # scoring dataset names
    set_names = ['threshold', 'test'] if 'y_threshold' in data_dict else ['test']

    # evaluation DataFrame for each considered dataset
    set_evaluation_dfs = []
    for n in set_names:
        # setup column space and hierarchical index for the current dataset
        periods_labels, periods_scores = data_dict[f'y_{n}'], data_dict[f'{n}_scores']
        column_names = get_column_names(periods_labels, event_types, ['PR_AUC'], n)
        set_evaluation_dfs.append(
            pd.DataFrame(
                columns=column_names, index=pd.MultiIndex.from_tuples([], names=['method', 'granularity'])
            )
        )
        evaluation_df = set_evaluation_dfs[-1]

        # add metrics when considering all traces and applications the same
        evaluation_df.loc[(config_name, 'global'), :] = get_metrics_row(
            periods_labels, periods_scores, evaluator, column_names, event_types,
            granularity='global', method_path=method_path, evaluation_string=evaluation_string,
            metrics_colors=metrics_colors
        )

        # if using spark data, add metrics for each application and trace
        periods_info = data_dict[f'{n}_info']
        if 'benchmark_userclicks' in periods_info[0][0]:
            # application-wise performance
            app_ids = set([int(info[0].split('_')[2]) for info in periods_info])
            for app_id in app_ids:
                app_indices = [i for i, info in enumerate(periods_info) if int(info[0].split('_')[2]) == app_id]
                app_labels, app_scores, app_info = [], [], []
                for i in range(len(periods_info)):
                    if i in app_indices:
                        app_labels.append(periods_labels[i])
                        app_scores.append(periods_scores[i])
                        app_info.append(periods_info[i])
                evaluation_df.loc[(config_name, f'app{app_id}'), :] = get_metrics_row(
                    app_labels, app_scores, evaluator, column_names, event_types, granularity='app',
                    metrics_colors=metrics_colors, method_path=method_path, periods_key=f'app{app_id}'
                )
                # trace-wise performance
                trace_ids = set([info[0] for info in app_info])
                for trace_id in trace_ids:
                    trace_indices = [i for i, info in enumerate(app_info) if info[0] == trace_id]
                    trace_labels, trace_scores, trace_info = [], [], []
                    for j in range(len(app_info)):
                        if j in trace_indices:
                            trace_labels.append(app_labels[j])
                            trace_scores.append(app_scores[j])
                            trace_info.append(app_info[j])
                    evaluation_df.loc[(config_name, trace_info[0][0]), :] = get_metrics_row(
                        trace_labels, trace_scores, evaluator, column_names, event_types,
                        granularity='trace', method_path=method_path, periods_key=trace_id
                    )
                # average performance across traces (in-trace separation ability)
                trace_rows = evaluation_df.loc[evaluation_df.index.get_level_values('granularity').str.contains('csv')]
                evaluation_df.loc[(config_name, 'trace_avg'), :] = trace_rows.mean(axis=0)
            # average performance across applications (in-application separation ability)
            app_rows = evaluation_df.loc[evaluation_df.index.get_level_values('granularity').str.contains('app')]
            evaluation_df.loc[(config_name, 'app_avg'), :] = app_rows.mean(axis=0)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    evaluation_df = pd.concat(set_evaluation_dfs, axis=1)
    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=[0, 1]).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        for index_key in evaluation_df.index:
            comparison_df.loc[index_key, :] = evaluation_df.loc[index_key, :].values
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')
