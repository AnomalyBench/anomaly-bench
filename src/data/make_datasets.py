"""AD Pipeline `train[/threshold]/test` datasets constitution module.
"""
import os
import importlib

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import parsers, get_output_path
from data.helpers import get_resampled
from data.data_managers import get_management_classes

if __name__ == '__main__':
    # get command line arguments
    args = parsers['make_datasets'].parse_args()

    # set output path
    OUTPUT_PATH = get_output_path(args, 'make_datasets')

    # set data manager depending on the input data
    data_manager = get_management_classes()[args.data](args)

    # load period DataFrames, labels and information
    DATA_PATHS_DICT = importlib.import_module(f'utils.{args.data}').DATA_PATHS_DICT
    period_dfs, labels, periods_info = data_manager.load_raw_data(DATA_PATHS_DICT)

    # add an `Anomaly` column to the period DataFrames based on the labels
    period_dfs = data_manager.add_anomaly_column(period_dfs, labels, periods_info)

    # handle NaN values in the raw period DataFrames
    period_dfs, periods_info = data_manager.get_handled_nans(period_dfs, periods_info)

    # resample period DataFrames to a new sampling period to save space if it is different from the original
    if args.pre_sampling_period != importlib.import_module(f'utils.{args.data}').SAMPLING_PERIOD:
        period_dfs = get_resampled(period_dfs, args.pre_sampling_period, anomaly_col=True)

    # save AD pipeline `train[/threshold]/test` period DataFrames and information
    data_manager.save_pipeline_datasets(period_dfs, periods_info, OUTPUT_PATH)
