"""OTN-specific modeling split module.
"""
import os

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_aligned_shuffle
from modeling.data_splitters import DataSplitter


class ServiceSplitter(DataSplitter):
    """Service splitting class.

    The `train/val/test` datasets are constituted from samples that belong to different (random) services.

    If we train on 80% of data, we randomly pick a set of services representing 80% of data and
    dedicate their periods to `train`.
    """
    def __init__(self, args, random_seed=26):
        super().__init__(args, random_seed)

    def custom_modeling_split(self, periods, periods_info, are_targets, sampling_f, **sampling_args):
        # samples (and possibly targets) to be returned
        set_names = ['train', 'val', 'test']

        print('grouping periods of same services into inner lists...', end=' ', flush=True)
        service_nos = set([t[0] for t in periods_info])
        grouped_periods = []
        for service_no in service_nos:
            grouped_periods.append([p for i, p in enumerate(periods) if periods_info[i][0] == service_no])
        grouped_periods = np.array(grouped_periods, dtype=object)
        print('done.')

        print('shuffling groups and computing val/test indices...', end=' ', flush=True)
        # shuffle the service groups
        np.random.shuffle(grouped_periods)
        # compute the cumulative number of records as we add service groups
        cum_group_records = np.cumsum(np.array([sum([p.shape[0] for p in g]) for g in grouped_periods]))
        # deduce the inclusive validation and test indices on the groups
        tot_records = np.concatenate(periods).shape[0]
        train_records = int((1 - self.test_prop - self.val_prop) * tot_records)
        val_records = int((1 - self.test_prop) * tot_records)
        val_idx = np.where(cum_group_records >= train_records)[0][0] + 1
        test_idx = np.where(cum_group_records >= val_records)[0][0] + 1
        print(f'done. (n_groups={len(cum_group_records)}, val_idx={val_idx}, test_idx={test_idx})')
        print('constituting train/val/test periods...', end=' ', flush=True)
        set_groups, set_periods = dict(), dict()
        for n, slice_ in zip(set_names, [slice(val_idx), slice(val_idx, test_idx), slice(test_idx, None)]):
            # service groups of the `n` dataset
            set_groups[n] = grouped_periods[slice_, ...]
            # deduced periods of the `n` dataset
            set_periods[n] = np.array([p for g in set_groups[n] for p in g], dtype=object)
        print('done.')
        # create and shuffle samples (and possibly targets) of each dataset
        datasets = dict()
        t = ' and targets' if are_targets else ''
        for n in set_periods:
            print(f'constituting {n} samples{t}...', end=' ', flush=True)
            samples, targets = np.array([]), (np.array([]) if are_targets else None)
            for period in set_periods[n]:
                p_items = sampling_f(period, **sampling_args)
                p_samples = p_items[0] if are_targets else p_items
                samples = np.concatenate([samples, p_samples]) if samples.size != 0 else p_samples
                if are_targets:
                    targets = np.concatenate([targets, p_items[1]]) if targets.size != 0 else p_items[1]
            shuffled = get_aligned_shuffle(samples, targets if are_targets else None)
            samples = shuffled[0] if are_targets else shuffled
            datasets[f'X_{n}'] = samples
            if are_targets:
                datasets[f'y_{n}'] = shuffled[1]
            print('done.')
        return datasets
