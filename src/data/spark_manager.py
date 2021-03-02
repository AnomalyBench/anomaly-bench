"""Spark-specific data management module.
"""
import os

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.data_managers import DataManager
from utils.common import get_dataset_names
from utils.spark import PATH_EXTENSIONS, EVENT_TYPES, file_app


def load_trace(trace_path):
    """Loads a Spark trace as a pd.DataFrame from its full input path.

    Args:
        trace_path (str): full path of the trace to load.

    Returns:
        pd.DataFrame: the trace indexed by time, with columns processed to be consistent between traces.
    """
    # load trace DataFrame with time as its converted datetime index
    trace_df = pd.read_csv(trace_path)
    trace_df.index = pd.to_datetime(trace_df['t'], unit='s')
    trace_df = trace_df.drop('t', axis=1)

    # remove the file prefix from the streaming metrics in order for their name to be consistent between traces
    trace_df.columns = [col.replace(os.path.basename(trace_path)[:-4] + '_', '') for col in trace_df.columns]

    # return the DataFrame with sorted columns (they might be in a different order depending on the file)
    return trace_df.reindex(sorted(trace_df.columns), axis=1)


class SparkManager(DataManager):
    """Spark-specific data management class.
    """
    def __init__(self, args):
        super().__init__(args)
        # restrict traces to a specific application id or trace types
        self.app_id = args.app_id
        self.trace_types = args.trace_types

    def load_raw_data(self, data_paths_dict):
        """Spark-specific raw data loading.

        - The loaded period as the application(s) traces.
        - The labels as a single spreadsheet DataFrame for all traces gathering all events information.
        - The periods information as initialized lists of the form `[file_name, trace_type]`.
        """
        period_dfs, periods_info = [], []
        print('loading ground-truth spreadsheet...', end=' ', flush=True)
        labels_path = os.path.join(data_paths_dict['labels'], 'ground_truth.csv')
        labels = pd.read_csv(labels_path)
        # convert timestamps to datetime format (`extended_pt_effect_end` is not used for now)
        for c in ['event_start', 'event_end', 'extended_end', 'extended_pt_effect_end']:
            labels[c] = pd.to_datetime(labels[c], unit='s')
        print('done.')
        take_all = self.app_id == 0
        # filter the keys so as to exclude labels and unwanted trace types
        excluded_keys = ['labels']
        if self.trace_types != '.':
            excluded_keys += list(set(PATH_EXTENSIONS.keys()) - set(self.trace_types.split('.')))
        period_dfs = []
        for type_, path in {k: v for k, v in data_paths_dict.items() if k not in excluded_keys}.items():
            app_text = 'all applications' if take_all else f'application {self.app_id}'
            print(f'loading {type_} traces of {app_text}...', end=' ', flush=True)
            # keep only files of the relevant application id if specified (never use apps #9 and #11)
            file_names = [
                fn for fn in os.listdir(path)
                if file_app(fn) not in [9, 11] and (take_all or file_app(fn) == self.app_id)
            ]
            for fn in file_names:
                period_dfs.append(load_trace(os.path.join(path, fn)))
                periods_info.append([fn, type_])
            print('done.')
        assert len(period_dfs) > 0, 'no traces to load for the provided app id and excluded trace types.'
        return period_dfs, labels, periods_info

    def add_anomaly_column(self, period_dfs, labels, periods_info):
        """Spark-specific `Anomaly` column extension.

        Note: `periods_info` is assumed to be of the form `[file_name, trace_type]`.

        `Anomaly` will be set to 0 if the record is outside any event range, otherwise it will be
        set to another value depending on the range type (as defined by utils.spark.EVENT_TYPES).
        => The label for a given range type corresponds to its index in the EVENT_TYPES list +1.
        """
        print('adding an `Anomaly` column to the Spark traces...', end=' ', flush=True)
        extended_dfs = [period_df.copy() for period_df in period_dfs]
        for i, period_df in enumerate(extended_dfs):
            period_df['Anomaly'] = 0
            file_name, trace_type = periods_info[i]
            if trace_type != 'undisturbed':
                for e_t in labels[labels['trace_id'] == file_name[:-4]].itertuples():
                    e_start = e_t.event_start
                    # either set the event end to the end of the root cause or extended effect if the latter is set
                    e_end = e_t.event_end if pd.isnull(e_t.extended_end) else e_t.extended_end
                    # set the label of an event type as its index in the types list +1
                    period_df.loc[(period_df.index >= e_start) & (period_df.index <= e_end), 'Anomaly'] = \
                        EVENT_TYPES.index(e_t.event_type) + 1
        print('done.')
        return extended_dfs

    def get_handled_nans(self, period_dfs, periods_info):
        """Spark-specific handling of NaN values.

        The only NaN values that were recorded were for inactive executors, found equivalent
        of them being -1.
        """
        print('handling NaN values encountered in Spark traces...', end=' ', flush=True)
        handled_dfs = [service_df.copy() for service_df in period_dfs]
        for period_df in handled_dfs:
            period_df.fillna(-1, inplace=True)
        print('done.')
        return handled_dfs, periods_info

    def get_pipeline_split(self, period_dfs, periods_info):
        """Spark-specific AD pipeline `train[/threshold]/test` splitting.

        A `threshold` dataset will only be used if using supervised threshold
        selection on the outlier scores.

        Note: `periods_info` is assumed to be of the form `[file_name, trace_type]`.
        It will however be returned in the form `[file_name, trace_type, period_rank]`.
        Where `period_rank` is the chronological rank of the period in its file.

        - All undisturbed traces will typically be sent to `train`.
        - Using supervised threshold selection, half of the disturbed traces will typically
            be sent to `threshold` and the other half to `test`.
        - Using unsupervised threshold selection, all disturbed traces will typically
            be sent to `test` (the threshold will be selected using a subpart of `train`).
        """
        # get dataset names depending on threshold selection supervision
        set_names = get_dataset_names(self.threshold_supervision)
        datasets = {k: {set_name: [] for set_name in set_names} for k in ['dfs', 'info']}

        # all undisturbed traces being sent to `train` should be a common pattern
        print('constituting train periods...', end=' ', flush=True)
        undisturbed_ids = [i for i, info in enumerate(periods_info) if info[1] == 'undisturbed']
        for i, period_df in enumerate(period_dfs):
            if i in undisturbed_ids:
                datasets['dfs']['train'].append(period_df)
                # every period is alone (hence at rank 0) in its trace file
                datasets['info']['train'].append(periods_info[i] + [0])
        print('done.')

        # unsupervised ts - send all disturbed traces to `test`
        if self.threshold_supervision == 'unsupervised':
            print('constituting test periods...', end=' ', flush=True)
            disturbed_ids = [i for i, info in enumerate(periods_info) if info[1] != 'undisturbed']
            for i, period_df in enumerate(period_dfs):
                if i in disturbed_ids:
                    datasets['dfs']['test'].append(period_df)
                    # every period is alone (hence at rank 0) in its trace file
                    datasets['info']['test'].append(periods_info[i] + [0])
            print('done.')

        # supervised ts - configuration-specific `threshold`/`test` split of disturbed traces
        else:
            file_splits = []
            print('constituting threshold and test periods...', end=' ', flush=True)
            # application #14 only using undisturbed, stalled input and cpu contention traces
            if self.app_id == 14 and self.trace_types == 'undisturbed.stalled_input.cpu_contention':
                file_splits = [
                    # stalled input file (2 events in `threshold`, 2 in `test`)
                    ('benchmark_userclicks_14_20_10000_1000000_batch066_8_.csv', pd.to_datetime('2018-06-07 00:00:01')),
                    # cpu contention file (3 events in `threshold`, 3 in `test`)
                    ('benchmark_userclicks_14_19_10003_1000000_batch146_19_.csv', pd.to_datetime('2018-06-14 21:30:00'))
                ]
            # implement other configurations if needed
            # ...
            for file_split in file_splits:
                period_idx = [i for i, info in enumerate(periods_info) if info[0] == file_split[0]][0]
                period_df, period_info = period_dfs[period_idx], periods_info[period_idx]
                threshold_part = period_df.loc[period_df.index < file_split[1]]
                test_part = period_df.loc[period_df.index >= file_split[1]]
                for set_name, df in zip(['threshold', 'test'], [threshold_part, test_part]):
                    # first file period goes to `threshold`, second to `test`
                    period_rank = 0 if set_name == 'threshold' else 1
                    for k, item in zip(['dfs', 'info'], [df, period_info + [period_rank]]):
                        datasets[k][set_name].append(item)
            print('done.')
        # return the period DataFrames and information per dataset
        return datasets['dfs'], datasets['info']
