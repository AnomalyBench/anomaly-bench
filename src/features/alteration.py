"""Features engineering/alteration module.

Features alteration steps are typically gathered into what we call "alteration bundles".

We start with an empty "result" DataFrame for each period, and sequentially add columns to it based on the
steps described in the considered bundle.

A bundle step is of the form `input_features: alteration_chain`. It projects the original period DataFrame
on `input_features` only, applies `alteration_chain` to them and adds the output columns to the result DataFrame.

`input_features` (tuple|str): Either a tuple of feature names or `all` if we want to consider
    all the original input features.

`alteration_chain` (str): dot-separated alteration functions to apply to the input features, where
    the output columns of a function constitute the input of the next.
    Each alteration function must be of the form `fname_arg1_arg2_..._argn`, where
    - `fname` refers to the function name, as defined in the `get_alteration_functions` function.
        If the name consists of multiple words, they should typically be merged without spacing.
    - `arg{i}` are the function's arguments, `_` separated and whose number/order have to match
        the function definition.

Note: alteration functions do not assume any `Anomaly` column present in the input DataFrames.
"""
import os
import functools

from tqdm import tqdm
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from features.tsfel.features_options import get_feature_functions


def get_altered_features(datasets, alteration_bundle):
    """Returns the provided datasets with their features altered by `alteration_bundle`.

    Args:
        datasets (dict): datasets of the form `{set_name: period_dfs}`.
        alteration_bundle (dict): features alteration bundle dictionary, as described above.

    Returns:
        dict: the datasets with altered features, in the same format as provided.
    """
    altered_datasets = dict()
    for set_name in datasets:
        print(f'altering features of {set_name} periods:')
        altered_datasets[set_name] = []
        # sequentially apply the alteration bundle to all periods of the dataset
        for period_df in tqdm(datasets[set_name]):
            altered_datasets[set_name].append(apply_alteration_bundle(period_df, alteration_bundle))
    return altered_datasets


def apply_alteration_bundle(period_df, alteration_bundle):
    """Returns `period_df` with its features altered by the provided alteration bundle.

    Args:
        period_df (pd.DataFrame): input period DataFrame whose features to alter.
        alteration_bundle (dict): alteration bundle dictionary to apply.

    Returns:
        pd.DataFrame: the same period with its features altered by the bundle.
    """
    # if any, remove the period's `Anomaly` column from consideration
    clean_df = period_df.copy()
    if 'Anomaly' in period_df:
        clean_df = period_df.drop('Anomaly', axis=1)

    # get alteration functions dictionary
    alteration_f_dict = get_alteration_functions()

    # sequentially add the outputs of the bundle's alteration steps to an empty result DataFrame
    result_df = pd.DataFrame()
    for input_features, alteration_chain in alteration_bundle.items():
        # consider all features if `input_features` is `all`
        input_features = slice(None) if input_features == 'all' else list(input_features)
        # constitute the chain output by sequentially calling its functions with their arguments
        chain_output_df = clean_df[input_features]
        for alteration_step in alteration_chain.split('.'):
            alteration_specs = alteration_step.split('_')
            chain_output_df = alteration_f_dict[alteration_specs[0]](chain_output_df, *alteration_specs[1:])
        # add the chain output to the period (only keeping rows for which it provided values)
        result_df = result_df.assign(**chain_output_df).dropna()

    # add back any `Anomaly` column to the records that were not dropped in the process (implicit join)
    if 'Anomaly' in period_df:
        result_df['Anomaly'] = period_df['Anomaly']
    return result_df


def get_alteration_functions():
    """Returns a dictionary gathering both common and data-specific features alteration functions.

    A getter function is used for solving cross-import issues.
    """
    # add absolute src directory to python path to import other project modules
    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from features.spark_alteration import add_executors_avg, add_nodes_avg
    from features.otn_alteration import add_network_gains
    return {
        'identity': apply_identity,
        'difference': add_differencing,
        'gains': add_network_gains,
        'execavg': add_executors_avg,
        'nodeavg': add_nodes_avg,
        'window': add_window_features
    }


def apply_identity(period_df):
    """Simply returns `period_df` without altering its features.
    """
    return period_df


def add_differencing(period_df, diff_factor_str, original_treatment):
    """Adds features differences, either keeping or dropping the original ones.

    Args:
        period_df (pd.DataFrame): input period DataFrame.
        diff_factor_str (str): differencing factor as a string integer.
        original_treatment (str): either `keep` or `drop`, specifying what to do with original features.

    Returns:
        pd.DataFrame: the input DataFrame with differenced features, with or without the original ones.
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'
    # apply differencing and drop records with NaN values
    difference_df = period_df.diff(int(diff_factor_str)).dropna()
    difference_df.columns = [f'{diff_factor_str}_diff_{c}' for c in difference_df.columns]
    # prepend original input features if we choose to keep them (implicit join if different counts)
    if original_treatment == 'keep':
        difference_df = pd.concat([period_df, difference_df], axis=1)
    return difference_df


def add_window_features(period_df, option_idx, window_type, original_treatment,
                        window_size=-1):
    """Adds window aggregate features, either keeping or dropping the original ones.

    Args:
        period_df (pd.DataFrame): input period DataFrame.
        option_idx (int): option index in `tsfel.features_options.FEATURES_OPTIONS`,
            containing the name and parameters of the features to compute.
        window_type (str): `rolling` or `expanding` depending on the type of window to use.
        original_treatment (str): either `keep` or `drop`, specifying what to do with original features.
        window_size (str): optional window size in datetime format (e.g. `3d`).
            Only considered (and required) for rolling windows.

    Returns:
        pd.DataFrame: the input DataFrame with window features, with or without the original ones.
    """
    # check parameters
    a_t = 'original features treatment can only be `keep` or `drop`'
    assert original_treatment in ['drop', 'keep'], a_t
    a_t = 'window type can only be `rolling` or `expanding`'
    assert window_type in ['rolling', 'expanding'], a_t
    a_t = 'a window size must be provided if extracting rolling window features'
    assert not (window_type == 'rolling' and window_size == -1), a_t
    # set windowing function based on the window type and window size
    if window_type == 'rolling':
        window_size = int(pd.Timedelta(window_size) / (period_df.index[1] - period_df.index[0]))
        windowing = functools.partial(period_df.rolling, window_size)
    else:
        windowing = functools.partial(period_df.expanding)
    # compute features excluding the current record for each window
    extracted_df = windowing().agg(get_feature_functions(int(option_idx))).shift(1)
    # flatten column names using the names of feature functions as suffixes
    extracted_df.columns = ['_'.join(c) for c in extracted_df.columns]
    # prepend original features if we choose to keep them (implicit join if different counts)
    if original_treatment == 'keep':
        extracted_df = pd.concat([period_df, extracted_df], axis=1)
    return extracted_df


def get_bundles_lists():
    """Returns a dictionary gathering the possible features alteration bundles lists.

    A getter function is used for solving cross-import issues.
    """
    # add absolute src directory to python path to import other project modules
    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from features.spark_alteration import SPARK_BUNDLES
    from features.otn_alteration import OTN_BUNDLES
    return {
        'spark_bundles': SPARK_BUNDLES,
        'otn_bundles': OTN_BUNDLES,
    }
