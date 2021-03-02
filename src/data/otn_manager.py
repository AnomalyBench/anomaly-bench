"""OTN-specific data management module.

TODO - add support for unsupervised threshold selection.
"""
import os

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.data_managers import DataManager


class OTNManager(DataManager):
    """OTN-specific data management class.
    """
    def __init__(self, args):
        super().__init__(args)
        # every `label` will span from `label + start_delta` to `label + end_delta`
        self.start_delta = pd.Timedelta(args.primary_start)
        self.end_delta = pd.Timedelta(args.tolerance_end)
        # number of services to consider (-1 meaning all)
        self.n_services = args.n_services

    def load_raw_data(self, data_paths_dict):
        """OTN-specific raw data loading.

        - The loaded periods are the service traces.
        - The labels are one label DataFrame per service trace, with the ground-truth timestamps as indices only.
        - The periods information are lists of the form `[service_no, service_type]`.
            `service_type` can be either 'anomalies' or 'no_anomalies', according to whether the service contains
            labeled anomalies or not.

        If not -1, we only consider `n_services` service traces.
        => We take 80% of this number as the last normal services, and 20% as
        the last anomalous services, to maintain the distribution of service types.
        Taking the last services was chosen over taking the first ones because it
        better preserves the temporal distribution of anomalies in services w/ alarms.
        """
        # check whether traces and labels relate to the same service numbers
        service_nos = dict()
        for data_type, data_path in data_paths_dict.items():
            service_nos[data_type] = sorted([int(fn[:-4]) for fn in os.listdir(data_path)])
        assert service_nos['traces'] == service_nos['labels'], \
            'traces and labels must relate to the same service numbers'

        # load the traces and labels sorted by service number
        data_dfs = dict()
        for data_type, data_path in data_paths_dict.items():
            print(f'loading service {data_type} from {data_path}...', end=' ', flush=True)
            data_dfs[data_type] = []
            for s in service_nos['traces']:
                full_path = os.path.join(data_path, f'{s}.csv')
                data_dfs[data_type].append(pd.read_csv(full_path, parse_dates=['Time'], index_col='Time'))
            print('done.')

        # derive the periods information
        def get_service_type(i, labels): return 'no_anomalies' if len(labels[i].index) == 0 else 'anomalies'
        periods_info = [[s, get_service_type(i, data_dfs['labels'])] for i, s in enumerate(service_nos['traces'])]

        # return either all or the first `n_services` divided in both service types
        if self.n_services == -1:
            return data_dfs['traces'], data_dfs['labels'], periods_info
        counts = {
            'max_normal': int(0.8 * self.n_services), 'cur_normal': 0,
            'max_anomalous': int(0.2 * self.n_services), 'cur_anomalous': 0
        }
        item_keys = ['traces', 'labels', 'info']
        limited = {k: [] for k in item_keys}
        r_tr, r_la, r_in = reversed(data_dfs['traces']), reversed(data_dfs['labels']), reversed(periods_info)
        for items in zip(r_tr, r_la, r_in):
            type_key = 'normal' if items[2][1] == 'no_anomalies' else 'anomalous'
            if counts[f'cur_{type_key}'] < counts[f'max_{type_key}']:
                for k, item in zip(item_keys, items):
                    limited[k].append(item)
                counts[f'cur_{type_key}'] += 1
        # return the items back in chronological order (although it is not important)
        return list(reversed(limited['traces'])), list(reversed(limited['labels'])), list(reversed(limited['info']))

    def add_anomaly_column(self, period_dfs, labels, periods_info):
        """OTN-specific `Anomaly` column extension.

        Here we do not have several anomaly types, so `Anomaly` is set to 0 if the record does not
        belong to any anomalous range, and set to 1 if it does.

        The anomalous range corresponding to a given point-based `label` is defined as:
        => `(label + start_delta, label + end_delta)`.
        `start_delta` and `end_delta` being positive or negative.
        """
        print('adding an `Anomaly` column to the service traces...', end=' ', flush=True)
        extended_dfs = [period_df.copy() for period_df in period_dfs]
        for i, period_df in enumerate(extended_dfs):
            period_df['Anomaly'] = 0
            for label in labels[i].index:
                period_df.loc[
                    (period_df.index >= label + self.start_delta) & (period_df.index <= label + self.end_delta),
                    'Anomaly'
                ] = 1
        print('done.')
        return extended_dfs

    def get_handled_nans(self, service_dfs, services_info):
        """OTN-specific handling of NaN values.

        For each service:
        - Removes leading 'all NaNs' records if any.
        - Cuts it in 2 parts, removing the period from `2018-05-24` to `2018-06-10` (both included)
            (in which there are 70% NaNs).
        - Drops the remaining periods having at least 30% NaN values for at least one feature
            (1.5% of data of services w/o anomalies).
        - Fills the remaining NaN values using the forward fill, backward fill and mean strategies
            (in this order).

        Service information is then replaced with periods information, of the form
        `[service_name, service_type, period_rank]`. Where `period_rank` is the chronological
        rank of the period in its service (starting at 0).
        """
        print('handling NaN values encountered in service traces...', end=' ', flush=True)
        period_dfs = [service_df.copy() for service_df in service_dfs]
        periods_info = services_info.copy()
        for i, service_df in enumerate(period_dfs):
            # remove rows until the first row that is not fully NaN (relevant for 11 services)
            service_df = service_df[service_df[service_df.notna().all(axis=1)].index[0]:]

            # remove all data from May 24th to June 10th, both included, where there are 70% NaN values
            removal_start, removal_end = pd.to_datetime('2018-05-24'), pd.to_datetime('2018-06-10')
            # each service is now constituted by 2 contiguous periods
            first_period = service_df.loc[service_df.index < removal_start].copy()
            second_period = service_df.loc[service_df.index > removal_end].copy()
            period_dfs[i] = [first_period, second_period]
            # extend the period information with the period rank in the service
            periods_info[i] = [periods_info[i] + [0], periods_info[i] + [1]]

            # fill the remaining NaN values in both contiguous periods
            for j, period_df in enumerate(period_dfs[i]):
                # if more than 30% NaNs for at least one feature, assign None to the period DataFrame and info
                if (100 * period_dfs[i][j].isnull().sum() / len(period_dfs[i][j])).max() >= 30.0:
                    period_dfs[i][j], periods_info[i][j] = None, None
                    # if the service's first period is removed, the second one becomes first
                    if j == 0:
                        periods_info[i][1][2] = 0
                else:
                    period_dfs[i][j] = period_df.fillna(method='ffill').fillna(method='bfill').fillna(period_df.mean())
        # return flattened period DataFrames and information removing None values
        flattened = dict()
        for k, full_list in zip(['dfs', 'info'], [period_dfs, periods_info]):
            flattened[k] = [item for service_list in full_list for item in service_list if item is not None]
        print('done.')
        return flattened['dfs'], flattened['info']

    def get_pipeline_split(self, period_dfs, periods_info):
        """OTN-specific AD pipeline `train/threshold/test` splitting.

        Note: `periods_info` is assumed to be of the form `[service_name, service_type, period_rank]`.
        Where `period_rank` is the chronological rank of the period in its service.

        - Periods from services without labeled anomalies go to `train` (79% of data).
        - The first periods of every other service until May 2018 (excluded) go to `threshold`.
        - The end of the first periods and the second periods go to `test`.
        """
        datasets = {k: {set_name: [] for set_name in ['train', 'threshold', 'test']} for k in ['dfs', 'info']}

        print('constituting train periods...', end=' ', flush=True)
        no_anomalies_ids = [i for i, info in enumerate(periods_info) if info[1] == 'no_anomalies']
        for k, items in zip(['dfs', 'info'], [period_dfs, periods_info]):
            datasets[k]['train'] = [item for i, item in enumerate(items) if i in no_anomalies_ids]
        print('done.')

        print('constituting threshold and test periods...', end=' ', flush=True)
        remaining = dict()
        for k, items in zip(['dfs', 'info'], [period_dfs, periods_info]):
            remaining[k] = [item for i, item in enumerate(items) if i not in no_anomalies_ids]
        for i, period_df in enumerate(remaining['dfs']):
            # leading periods are further split into `threshold` and `test` at May 2018
            if remaining['info'][i][2] == 0:
                threshold_part = period_df.loc[period_df.index < pd.to_datetime('2018-05')]
                test_part = period_df.loc[period_df.index >= pd.to_datetime('2018-05')]
                for set_name, df in zip(['threshold', 'test'], [threshold_part, test_part]):
                    if not df.empty:
                        # the split going to test is the new second period of the service
                        updated_info = remaining['info'][i].copy()
                        updated_info[2] = 0 if set_name == 'threshold' else 1
                        for k, item in zip(['dfs', 'info'], [df, updated_info]):
                            datasets[k][set_name].append(item)
            # second periods entirely go to `test`
            else:
                # since the first periods were split, the second periods are now third in the service
                remaining['info'][i][2] = 2
                for k, item in zip(['dfs', 'info'], [period_df, remaining['info'][i]]):
                    datasets[k]['test'].append(item)
        print('done.')
        # return the period DataFrames and information per dataset
        return datasets['dfs'], datasets['info']
