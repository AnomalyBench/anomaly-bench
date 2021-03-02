"""Final anomaly detection evaluation module.
"""
import os

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from scoring.evaluation import get_column_names


def get_metrics_row(labels, preds, evaluator, column_names, event_types, granularity):
    """Returns the metrics row to add to the evaluation DataFrame.

    Args:
        labels (ndarray): periods labels of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        preds (ndarray): periods binary predictions with the same shape as `labels`.
        evaluator (metrics.Evaluator): object defining the binary metrics of interest.
        column_names (list): list of column names corresponding to the metrics to compute.
        event_types (list|None): names of the event types that we might encounter in the data (if relevant).
        granularity (str): evaluation granularity.
            Must be either `global`, for overall, `app`, for app-wise or `trace`, for trace-wise.

    Returns:
        list: list of metrics to add to the evaluation DataFrame (corresponding to `column_names`).
    """
    assert granularity in ['global', 'app', 'trace']

    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)
    metric_names = ['PRECISION', 'RECALL', f'F{evaluator.beta}_SCORE']

    # recover dataset name, prefix and title from the column names
    set_prefix = f'{column_names[0].split("_")[0]}_'
    if set_prefix not in ['THRESHOLD_', 'TEST_']:
        set_prefix = ''

    # compute the metrics defined by the evaluator
    f, p, r = evaluator.compute_metrics(labels, preds)
    # we do not use average metrics across types here
    for d in f, r:
        d.pop('avg')
    # case of a single event type
    if event_types is None:
        for m_name, m_value in zip(metric_names, [p, r['global'], f['global']]):
            metrics_row.at[0, f'{set_prefix}{m_name}'] = m_value
    # case of multiple known event types
    else:
        # `global` and interpretable event types that belong to the keys of recall curves
        class_names = ['global'] + [event_types[i-1] for i in range(1, len(event_types) + 1) if i in r]
        # add the metric and column corresponding to each type
        for cn in class_names:
            if cn == 'global':
                for m_name, m_value in zip(metric_names, [p, r['global'], f['global']]):
                    metrics_row.at[0, f'{set_prefix}GLOBAL_{m_name}'] = m_value
            else:
                # precision is only considered globally
                class_metric_names = [m for m in metric_names if m != 'PRECISION']
                label_key = event_types.index(cn) + 1
                for m_name, m_value in zip(class_metric_names, [r[label_key], f[label_key]]):
                    metrics_row.at[0, f'{set_prefix}{cn.upper()}_{m_name}'] = m_value
    return metrics_row.iloc[0].tolist()


def save_detection_evaluation(data_dict, evaluator, evaluation_string, config_name, spreadsheet_path,
                              *, event_types=None, metrics_colors=None, method_path=None):
    """Adds the detection evaluation of the provided `configuration` to a sorted comparison spreadsheet.

    If a list of types is passed as `event_types`, the evaluation will be both performed "globally"
    (i.e. considering all types together) and type-wise.
    The passed `event_types` can be a subset of the types that are indeed represented in the data,
    i.e. whose `index + 1` appear in the labels.

    If `event_types` is None, the labels will be considered binary. Only the "global" performance
    will be reported and considered as default.

    Note: the term `global` is both used when making no difference between applications or traces
    ("global granularity") and when making no difference between event types ("global type").

    Args:
        data_dict (dict): [threshold and] test periods record-wise outlier predictions, labels and information:
            - predictions as `(set_name)_(preds): ndarray`.
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
        method_path (str): thresholding method path, to save any extended evaluation to.
    """
    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{evaluation_string}_detection_comparison.csv')

    # thresholding dataset names
    set_names = ['threshold', 'test'] if 'y_threshold' in data_dict else ['test']

    # evaluation DataFrame for each considered dataset
    set_evaluation_dfs = []
    for n in set_names:
        # setup column space and hierarchical index for the current dataset
        periods_labels, periods_preds = data_dict[f'y_{n}'], data_dict[f'{n}_preds']
        column_names = get_column_names(
            periods_labels, event_types,
            ['PRECISION', 'RECALL', f'F{evaluator.beta}_SCORE'], n
        )
        set_evaluation_dfs.append(
            pd.DataFrame(
                columns=column_names, index=pd.MultiIndex.from_tuples([], names=['method', 'granularity'])
            )
        )
        evaluation_df = set_evaluation_dfs[-1]

        # add metrics when considering all traces and applications the same
        evaluation_df.loc[(config_name, 'global'), :] = get_metrics_row(
            periods_labels, periods_preds, evaluator, column_names, event_types, granularity='global'
        )

        # if using spark data, add metrics for each application and trace
        periods_info = data_dict[f'{n}_info']
        if 'benchmark_userclicks' in periods_info[0][0]:
            # application-wise performance
            app_ids = set([int(info[0].split('_')[2]) for info in periods_info])
            for app_id in app_ids:
                app_indices = [i for i, info in enumerate(periods_info) if int(info[0].split('_')[2]) == app_id]
                app_labels, app_preds, app_info = [], [], []
                for i in range(len(periods_info)):
                    if i in app_indices:
                        app_labels.append(periods_labels[i])
                        app_preds.append(periods_preds[i])
                        app_info.append(periods_info[i])
                evaluation_df.loc[(config_name, f'app{app_id}'), :] = get_metrics_row(
                    app_labels, app_preds, evaluator, column_names, event_types, granularity='app'
                )
                # trace-wise performance
                trace_ids = set([info[0] for info in app_info])
                for trace_id in trace_ids:
                    trace_indices = [i for i, info in enumerate(app_info) if info[0] == trace_id]
                    trace_labels, trace_preds, trace_info = [], [], []
                    for j in range(len(app_info)):
                        if j in trace_indices:
                            trace_labels.append(app_labels[j])
                            trace_preds.append(app_preds[j])
                            trace_info.append(app_info[j])
                    evaluation_df.loc[(config_name, trace_info[0][0]), :] = get_metrics_row(
                        trace_labels, trace_preds, evaluator, column_names, event_types, granularity='trace'
                    )
                # average performance across traces (in-trace detection ability)
                trace_rows = evaluation_df.loc[evaluation_df.index.get_level_values('granularity').str.contains('csv')]
                evaluation_df.loc[(config_name, 'trace_avg'), :] = trace_rows.mean(axis=0)
            # average performance across applications (in-application detection ability)
            app_rows = evaluation_df.loc[evaluation_df.index.get_level_values('granularity').str.contains('app')]
            evaluation_df.loc[(config_name, 'app_avg'), :] = app_rows.mean(axis=0)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    evaluation_df = pd.concat(set_evaluation_dfs, axis=1)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
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
