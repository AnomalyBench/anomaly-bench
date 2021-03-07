"""Data helpers module.

Gathers functions for loading and manipulating period DataFrames and numpy arrays.
"""
import os
import pickle

import numpy as np
import pandas as pd


def get_resampled(period_dfs, sampling_period, agg='mean', anomaly_col=False):
    """Returns the period DataFrames resampled to a new sampling period using the provided aggregation.

    Args:
        period_dfs (list): the list of input period DataFrames.
        sampling_period (str): the new sampling period, as a valid argument to `pd.Timedelta`.
        agg (str): the aggregation function for the resampling process (e.g. `mean` or `median`).
        anomaly_col (bool): whether the provided DataFrames have an `Anomaly` column, to resample differently.

    Returns:
        list: the same periods with resampled records.
    """
    resampled_dfs, sampling_p = [], pd.Timedelta(sampling_period)
    feature_cols = [c for c in period_dfs[0].columns if c != 'Anomaly']
    print(f'resampling periods applying records `{agg}` every {sampling_period}...', end=' ', flush=True)
    for df in period_dfs:
        resampled_df = df[feature_cols].resample(sampling_p).agg(agg).ffill().bfill()
        if anomaly_col:
            # if no records during `sampling_p`, we choose to simply repeat the label of the last one here
            resampled_df['Anomaly'] = df['Anomaly'].resample(sampling_p).agg('max').ffill().bfill()
        resampled_dfs.append(resampled_df)
    print('done.')
    return resampled_dfs


def load_files(input_path, file_names, file_format, *,
               drop_info_suffix=False, drop_labels_prefix=False):
    """Loads and returns the provided `file_names` files from `input_path`.

    Args:
        input_path (str): path to load the files from.
        file_names (list): list of file names to load (without file extensions).
        file_format (str): format of the files to load, must be either `pickle` or `numpy`.
        drop_info_suffix (bool): if True, drop any `_info` string from the output dict keys.
        drop_labels_prefix (bool): if True, drop any `y_` string from the output dict keys.

    Returns:
        dict: the loaded files, with as keys the file names.
    """
    assert file_format in ['pickle', 'numpy'], 'supported format only include `pickle` and `numpy`'
    ext = 'pkl' if file_format == 'pickle' else 'npy'
    files_dict = dict()
    print(f'loading {file_format} files from {input_path}')
    for fn in file_names:
        print(f'loading `{fn}.{ext}`...', end=' ', flush=True)
        if file_format == 'pickle':
            files_dict[fn] = pickle.load(open(os.path.join(input_path, f'{fn}.{ext}'), 'rb'))
        else:
            files_dict[fn] = np.load(os.path.join(input_path, f'{fn}.{ext}'), allow_pickle=True)
        print('done.')
    if not (drop_info_suffix or drop_labels_prefix):
        return files_dict
    if drop_info_suffix:
        files_dict = {k.replace('_info', ''): v for k, v in files_dict.items()}
    if drop_labels_prefix:
        return {k.replace('y_', ''): v for k, v in files_dict.items()}
    return files_dict


def all_files_exist(input_path, file_names, file_format):
    """Returns True if all the files exist at the provided path, False otherwise.

    Args:
        input_path (str): root path to the files.
        file_names (list): list of file names to check (without file extensions).
        file_format (str): format of the files to check, must be either `pickle` or `numpy`.

    Returns:
        bool: True if all the files exist, False otherwise.
    """
    assert file_format in ['pickle', 'numpy'], 'supported format only include `pickle` and `numpy`'
    ext = 'pkl' if file_format == 'pickle' else 'npy'
    print(f'loading {file_format} files from {input_path}')
    for fn in file_names:
        if file_format == 'pickle':
            try:
                pickle.load(open(os.path.join(input_path, f'{fn}.{ext}'), 'rb'))
            except FileNotFoundError:
                return False
        else:
            try:
                np.load(os.path.join(input_path, f'{fn}.{ext}'), allow_pickle=True)
            except FileNotFoundError:
                return False
    return True


def save_files(output_path, files_dict, file_format):
    """Saves files from the provided `files_dict` to `output_path` in the relevant format.

    Args:
        output_path (str): path to save the files to.
        files_dict (dict): dictionary of the form {`file_name`: `item_to_save`} (file names without extensions).
        file_format (str): format of the files to save, must be either `pickle` or `numpy`.
    """
    assert file_format in ['pickle', 'numpy'], 'supported format only include `pickle` and `numpy`'
    ext = 'pkl' if file_format == 'pickle' else 'npy'
    print(f'saving {file_format} files to {output_path}')
    os.makedirs(output_path, exist_ok=True)
    for fn in files_dict:
        print(f'saving `{fn}.{ext}`...', end=' ', flush=True)
        if file_format == 'pickle':
            with open(os.path.join(output_path, f'{fn}.{ext}'), 'wb') as pickle_file:
                pickle.dump(files_dict[fn], pickle_file)
        else:
            np.save(os.path.join(output_path, f'{fn}.{ext}'), files_dict[fn], allow_pickle=True)
        print('done.')


def load_mixed_formats(file_paths, file_names, file_formats):
    """Loads and returns `file_names` stored as `file_formats` at `file_paths`.

    Args:
        file_paths (list): list of paths for each file name.
        file_names (list): list of file names to load (without file extensions).
        file_formats (list): list of file formats for each file name, must be either `pickle` or `numpy`.

    Returns:
        dict: the loaded files, with as keys the file names.
    """
    assert len(file_names) == len(file_paths) == len(file_formats), 'the provided lists must be of same lengths'
    files = dict()
    for name, path, format_ in zip(file_names, file_paths, file_formats):
        files[name] = load_files(path, [name], format_)[name]
    return files


def load_datasets_data(input_path, info_path, dataset_names):
    """Returns the periods records, labels and information for the provided dataset names.

    Args:
        input_path (str): input path from which to load the records and labels.
        info_path (str): input path from which to load the periods information.
        dataset_names (list): list of dataset names.

    Returns:
        dict: the datasets data, with keys of the form `{n}`, `y_{n}` and `{n}_info` (`n` the dataset name).
    """
    file_names = [fn for n in dataset_names for fn in [n, f'y_{n}', f'{n}_info']]
    n_sets = len(dataset_names)
    file_paths, file_formats = n_sets * (2 * [input_path] + [info_path]), n_sets * (2 * ['numpy'] + ['pickle'])
    return load_mixed_formats(file_paths, file_names, file_formats)


def extract_save_labels(period_dfs, labels_file_name, output_path):
    """Extracts and saves labels from the `Anomaly` columns of the provided period DataFrames.

    Args:
        period_dfs (list): list of period pd.DataFrame.
        labels_file_name (str): name of the numpy labels file to save (without `.npy` extension).
        output_path (str): path to save the labels to, as a numpy array of shape `(n_periods, n_records)`.
            Where `n_records` depends on the period.

    Returns:
        list: the input periods without their `Anomaly` columns.
    """
    new_periods, labels_list = [period_df.copy() for period_df in period_dfs], []
    for i, period_df in enumerate(period_dfs):
        labels_list.append(np.array(period_df['Anomaly']))
        new_periods[i].drop('Anomaly', axis=1, inplace=True)

    # save labels and return the periods without their `Anomaly` columns
    print(f'saving {labels_file_name} labels file...', end=' ', flush=True)
    np.save(os.path.join(output_path, f'{labels_file_name}.npy'), np.array(labels_list, dtype=object))
    print('done.')
    return new_periods


def get_numpy_from_dfs(period_dfs):
    """Returns the equivalent numpy 3d-array for the provided list of period DataFrames.

    Args:
        period_dfs (list): the list of period DataFrames to turn into a numpy array.

    Returns:
        ndarray: corresponding numpy array of shape `(n_periods, period_size, n_features)`.
            Where `period_size` depends on the period.
    """
    return np.array([period_df.values for period_df in period_dfs], dtype=object)


def get_aligned_shuffle(array_1, array_2=None):
    """Returns `array_1` and `array_2` randomly shuffled, preserving alignments between the array elements.

    If `array_2` is None, simply return shuffled `array_1`.
    If it is not, the provided arrays must have the same number of elements.

    Args:
        array_1 (ndarray): first array to shuffle.
        array_2 (ndarray|None): second array to shuffle accordingly if not None.

    Returns:
        (ndarray, ndarray)|ndarray: the shuffled array(s).
    """
    assert array_2 is None or len(array_1) == len(array_2), 'arrays to shuffle must be have the same lengths'
    mask = np.random.permutation(len(array_1))
    if array_2 is None:
        return array_1[mask]
    return array_1[mask], array_2[mask]
