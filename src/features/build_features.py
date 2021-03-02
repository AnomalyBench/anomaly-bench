"""Features building module.

Turns the raw period variables into the final features that will be used by the detection models.
If specified, this pipeline step can also resample the periods to a new sampling period.
"""
import os

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import parsers, get_output_path, get_dataset_names
from data.helpers import load_files, save_files, all_files_exist, get_resampled, extract_save_labels, get_numpy_from_dfs
from features.alteration import get_altered_features, get_bundles_lists
from features.transformers import transformation_classes


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['build_features'].parse_args()

    # get input and output paths
    INPUT_DATA_PATH = get_output_path(args, 'make_datasets')
    OUTPUT_DATA_PATH = get_output_path(args, 'build_features', 'data')
    OUTPUT_MODELS_PATH = get_output_path(args, 'build_features', 'models')

    # load datasets (not repeating the feature extraction step if done before)
    set_names = get_dataset_names(args.threshold_supervision, disturbed_only=False)
    extracted_file_names = [f'{n}_extracted' for n in set_names]
    existing_features = all_files_exist(OUTPUT_DATA_PATH, extracted_file_names, 'pickle')
    if existing_features:
        print('loading existing extracted features.')
        datasets = load_files(OUTPUT_DATA_PATH, extracted_file_names, 'pickle')
        datasets = {k.replace('_extracted', ''): v for k, v in datasets.items()}
    else:
        datasets = load_files(INPUT_DATA_PATH, set_names, 'pickle')

    # optional features alteration bundle
    if args.alter_bundles != '.' and not existing_features:
        print(f'altering features using bundle #{args.alter_bundle_idx} of {args.alter_bundles}')
        datasets = get_altered_features(datasets, get_bundles_lists()[args.alter_bundles][args.alter_bundle_idx])
        # save data with extracted features to avoid repeating the step
        save_files(OUTPUT_DATA_PATH, {f'{k}_extracted': v for k, v in datasets.items()}, 'pickle')

    # resample periods of all datasets if the sampling period has changed from the previous one
    if args.sampling_period != args.pre_sampling_period:
        for k in datasets:
            datasets[k] = get_resampled(datasets[k], args.sampling_period, anomaly_col=True)

    # turn the `Anomaly` columns into numpy labels and drop them from the period DataFrames
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    print(f'saving labels into {OUTPUT_DATA_PATH}')
    for k in datasets:
        datasets[k] = extract_save_labels(datasets[k], f'y_{k}', OUTPUT_DATA_PATH)

    # turn datasets to `(n_periods, period_size, n_features)` ndarrays (`period_size` depends on the period)
    print('converting datasets to numpy arrays...', end=' ', flush=True)
    for k in datasets:
        datasets[k] = get_numpy_from_dfs(datasets[k])
    print('done.')

    # optional features transformation chain
    datasets_info = load_files(
        INPUT_DATA_PATH, [f'{n}_info' for n in set_names], 'pickle', drop_info_suffix=True
    )
    for transform_step in [ts for ts in args.transform_chain.split('.') if len(ts) > 0]:
        args_text = ''
        if 'scaling' in transform_step:
            args_text = f'{args.scaling_method}_'
        if 'pca' in transform_step:
            args_text = f'{args.n_components}_'
        if 'head' in transform_step:
            args_text = f'{args.head_size}_{args_text}'
        print(f'applying `{args_text}{transform_step}` to period features...', end=' ', flush=True)
        transformer = transformation_classes[transform_step](args, OUTPUT_MODELS_PATH)
        datasets = transformer.fit_transform_datasets(datasets, datasets_info)
        print('done.')

    # save periods with updated features
    save_files(OUTPUT_DATA_PATH, datasets, 'numpy')
