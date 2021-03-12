"""Data management module.

This module defines the methods that data-specific managers will have to implement for:
- Loading the raw data as period DataFrames with parsed dates.
- Labeling the periods based on the anomaly information.
- Splitting the periods into the pipeline's train[/threshold]/test` datasets.

A labeled `threshold` dataset will only be used if using supervised threshold selection
on the outlier scores.
"""
import os
import pickle
from abc import abstractmethod

import pandas as pd


class DataManager:
    """Data management base class.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
    """
    def __init__(self, args):
        # use either supervised or unsupervised threshold selection
        self.threshold_supervision = args.threshold_supervision

    @abstractmethod
    def load_raw_data(self, data_paths_dict):
        """Returns the loaded raw period DataFrames, labels and information.

        The period DataFrames will be loaded with timestamps as indices and the original features as columns.

        The labels can either be returned as a list of DataFrames (for each period) or as a single
        DataFrame containing all the anomalies information.

        Args:
            data_paths_dict (dict): input root paths. Keys are the paths description and values the actual paths.

        Returns:
            list, list|pd.DataFrame, list: the period DataFrames, labels and information.
        """

    @abstractmethod
    def add_anomaly_column(self, period_dfs, labels, periods_info):
        """Returns the period DataFrames with an additional `Anomaly` column, based on `labels`.

        Optionally uses periods information in deriving the labels.

        `Anomaly` should be set to:
        - 0 for the records outside any anomalous range,
        - 1 or any number reflecting the type of anomaly for the others.

        Args:
            period_dfs (list): periods to label as a list of pd.DataFrame.
            labels (list|pd.DataFrame): either a DataFrame per period or a single one gathering all anomalies.
            periods_info (list): periods information, as a list for each period of the corresponding index.

        Returns:
            list: the same period DataFrames augmented with the `Anomaly` column.
        """

    @abstractmethod
    def get_handled_nans(self, period_dfs, periods_info):
        """Returns the period DataFrames with handled NaN values.

        If a large portion of any given period contains a lot of NaN values, we allow this period to
        be split, removing that portion. The periods information list will however have to be updated
        accordingly to reflect the information of the new periods.

        Args:
            period_dfs (list): the list of input period DataFrames.
            periods_info (list): the corresponding periods information.

        Returns:
            list: the new period DataFrames and information, with handled NaN values.
        """

    def save_pipeline_datasets(self, period_dfs, periods_info, output_path):
        """Saves the AD pipeline datasets by splitting the input periods.

        - Periods in `train` will be modeled through a downstream task (e.g. forecasting).
        - If using supervised threshold selection, periods in `threshold` will be used
            for threshold selection on the outlier score.
        - Periods in `test` will be used for reporting the final performance of the AD model.

        Args:
            period_dfs (list): the list of input period DataFrames.
            periods_info (list): the corresponding periods information.
            output_path (str): path to save the dataset periods and information to.
        """
        periods_dict, info_dict = self.get_pipeline_split(period_dfs, periods_info)
        print(f'saving datasets into {output_path}')
        os.makedirs(output_path, exist_ok=True)
        for sn in periods_dict:
            print(f'saving {sn} period DataFrames and information...', end=' ', flush=True)
            for file_name, item in zip([sn, f'{sn}_info'], [periods_dict[sn], info_dict[sn]]):
                with open(os.path.join(output_path, f'{file_name}.pkl'), 'wb') as pickle_file:
                    pickle.dump(item, pickle_file)
            print('done.')

    @abstractmethod
    def get_pipeline_split(self, period_dfs, periods_info):
        """Returns the AD pipeline `train[/threshold]/test` period DataFrames and information.

        Args:
            period_dfs (list): the list of input period DataFrames.
            periods_info (list): the corresponding periods information.

        Returns:
            dict, dict: respectively the lists of period DataFrames and information for each dataset.
                The keys are the dataset names and the values the corresponding lists.
        """


# use a getter function to access references to data management classes to solve cross-import issues
def get_management_classes():
    """Returns a dictionary gathering references to the defined data management classes.
    """
    # add absolute src directory to python path to import other project modules
    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from data.spark_manager import SparkManager
    return {
        'spark': SparkManager
    }
