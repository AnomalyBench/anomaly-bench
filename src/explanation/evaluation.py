"""Explanation discovery evaluation module.
"""
import os

import numpy as np
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import get_explanation_evaluation_str


def get_explanation_samples(periods, periods_labels, min_sample_length=None, min_anomaly_length=None):
    """Returns the periods and periods labels formatted for explanation discovery grouped by anomaly type.

    The data is returned windowed, with each sample being of the form `(normal span + anomaly span)`.

    A sample is extracted per anomalous range, and consists in the anomaly interval preceded by
    the interval of normal data before it. If no normal data occurred before the beginning of an
    anomaly instance, we drop the corresponding sample.

    Optionally, samples containing less than `min_sample_length` records can be ignored, as well
    as samples containing anomalies having less than `min_anomaly_length` records.

    Args:
        periods (ndarray): periods records of shape `(n_periods, period_length, n_features)`.
            Where `period_length` depends on the period.
        periods_labels (ndarray): multiclass periods labels of the same shape.
        min_sample_length (int): optional minimum sample length in the returned samples.
        min_anomaly_length (int): optional minimum anomaly length in the returned samples.

    Returns:
        dict, dict: `(normal + anomaly)` samples and labels grouped by event type in the form
            {`event_type`: `samples|samples_labels`}.
    """
    pos_classes = np.delete(np.unique(np.concatenate(periods_labels, axis=0)), 0)
    samples = {pc: [] for pc in pos_classes}
    samples_labels = {pc: [] for pc in pos_classes}
    for period, period_labels in zip(periods, periods_labels):
        for pc in pos_classes:
            # type-specific label mask used for sample extraction
            label_mask = (np.append(period_labels, 0) == pc).astype(int)
            sample_start_ids = np.concatenate([
                [0],
                np.where(np.diff(label_mask) == -1)[0] + 1
            ])
            for i in range(len(sample_start_ids) - 1):
                sample_labels = period_labels[sample_start_ids[i]:sample_start_ids[i+1]]
                # only add samples that contain a normal prefix
                if 0 in sample_labels:
                    # optionally, only add samples with sufficient total length and anomaly length
                    if min_sample_length is None or len(sample_labels) >= min_sample_length:
                        anomaly_length = len(sample_labels[sample_labels > 0])
                        if min_anomaly_length is None or anomaly_length >= min_anomaly_length:
                            samples[pc].append(period[sample_start_ids[i]:sample_start_ids[i+1]])
                            samples_labels[pc].append(sample_labels)
    # convert to object if arrays of arrays of different sizes
    for pc in pos_classes:
        if len(samples_labels[pc]) != 1:
            samples[pc] = np.array(samples[pc], dtype=object)
            samples_labels[pc] = np.array(samples_labels[pc], dtype=object)
        else:
            samples[pc] = np.array(samples[pc])
            samples_labels[pc] = np.array(samples_labels[pc])
    return samples, samples_labels


def save_explanation_evaluation(args, data_dict, explainer, config_name, spreadsheet_path, event_types=None):
    """

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        data_dict (dict): evaluated periods and labels of the form
            `{set_name: periods, y_[set_name]: periods_labels}`.
        explainer (Explainer): evaluated explanation discovery object.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.
        event_types (list|None): names of the event types that we might encounter in the data (if relevant).
    """
    # set the full path for the comparison spreadsheet
    evaluation_task_str = get_explanation_evaluation_str(args)
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{evaluation_task_str}_comparison.csv')

    # explanation discovery dataset names
    set_names = ['threshold', 'test'] if 'y_threshold' in data_dict else ['test']

    # evaluation DataFrame for each considered dataset
    set_evaluation_dfs = []
    for n in set_names:
        # get periods and labels for the current dataset
        periods, periods_labels = data_dict[n], data_dict[f'y_{n}']

        # get explanation samples by event type
        samples_dict, samples_labels_dict = get_explanation_samples(
            periods, periods_labels, min_sample_length=args.exp_eval_min_sample_length,
            min_anomaly_length=args.exp_eval_min_anomaly_length
        )
        # add explanation metrics for the samples of the current dataset
        metrics_row = pd.DataFrame()
        for k in samples_dict:
            event_str = '' if event_types is None else f'{event_types[k-1].upper()}_'
            metrics_dict = explainer.fit_evaluate_samples(
                samples_dict[k], samples_labels_dict[k],
                test_prop=args.exp_eval_test_prop, n_runs=args.exp_eval_n_runs
            )
            for m_name in metrics_dict:
                metrics_row.at[config_name, f'{n.upper()}_{event_str}{m_name.upper()}'] = metrics_dict[m_name]
        set_evaluation_dfs.append(metrics_row)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    evaluation_df = pd.concat(set_evaluation_dfs, axis=1)
    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=0).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        for index_key in evaluation_df.index:
            comparison_df.loc[index_key, :] = evaluation_df.loc[index_key, :].values
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')
