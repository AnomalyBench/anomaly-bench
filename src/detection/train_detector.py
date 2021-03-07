"""Threshold selection module.

Threshold selection can either be `supervised` or `unsupervised`:
- Supervised: selects the outlier score threshold maximizing the AD performance
    on the (labeled) `threshold` dataset.
- Unsupervised: selects the outlier score threshold from the outlier scores distribution
    of the modeling's test samples (normal/unlabeled data).

To improve efficiency, the threshold selection method and parameters have to be provided
as lists of values to try (instead of single values).
"""
import os
import argparse
import importlib
import itertools

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    hyper_to_path, parsers, get_output_path, get_evaluation_string, get_thresholding_args,
    get_dataset_names, forecasting_choices, reconstruction_choices
)
from data.helpers import load_datasets_data, load_mixed_formats

from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from modeling.forecasting.forecasters import forecasting_classes
from modeling.reconstruction.reconstructors import reconstruction_classes

from detection.detector import Detector
from detection.evaluation import save_detection_evaluation

from visualization.helpers.spark import EVENT_TYPES
from metrics.evaluators import evaluation_classes


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['train_detector'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')
    MODEL_INPUT_PATH = get_output_path(args, 'train_model')
    SCORER_INPUT_PATH = get_output_path(args, 'train_scorer')
    OUTPUT_PATH = get_output_path(args, 'train_detector', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_detector', 'comparison')

    # load the periods, labels and information used to evaluate the final anomaly predictions
    thresholding_sets = get_dataset_names(args.threshold_supervision, disturbed_only=True)
    thresholding_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, thresholding_sets)

    # load the model and scoring classes for the relevant type of task
    a_m = 'supported models are limited to forecasting-based and reconstruction-based models'
    assert args.model_type in forecasting_choices + reconstruction_choices, a_m
    task_type = 'forecasting' if args.model_type in forecasting_choices else 'reconstruction'
    model_classes = forecasting_classes if task_type == 'forecasting' else reconstruction_classes
    scoring_classes = importlib.import_module(f'scoring.{task_type}.scorers').scoring_classes
    if args.model_type == 'naive.forecasting':
        model = model_classes[args.model_type](args, '')
    else:
        full_model_path = os.path.join(MODEL_INPUT_PATH, 'model.h5')
        model = model_classes[args.model_type].from_file(args, full_model_path)

    # initialize or load the relevant scorer from disk
    if args.scoring_method == 'nll':
        full_scorer_path = os.path.join(SCORER_INPUT_PATH, 'model.pkl')
        scorer = scoring_classes[args.scoring_method].from_file(args, model, full_scorer_path)
    else:
        scorer = scoring_classes[args.scoring_method](args, model, '')

    # adapt labels to cut out the first `n_back` records of each period if forecasting task
    if task_type == 'forecasting':
        for n in thresholding_sets:
            thresholding_data[f'y_{n}'] = get_trimmed_periods(thresholding_data[f'y_{n}'], args.n_back)

    # build the AD evaluator based on our definition of Precision and Recall
    evaluator = evaluation_classes[args.evaluation_type](args)

    # event types and metrics colors if relevant
    event_types = EVENT_TYPES if args.data == 'spark' else None
    metrics_colors = importlib.import_module(f'visualization.helpers.{args.data}').METRICS_COLORS

    # argument names for which to try every combination
    looped_arg_names = ['threshold_selection']
    args_dict = vars(args)

    # unsupervised threshold selection
    if args.threshold_supervision == 'unsupervised':
        # select the threshold from the scores distribution of the modeling's test samples
        print('loading training periods and information...', end=' ', flush=True)
        modeling_files = load_mixed_formats(
            [DATA_INPUT_PATH, DATA_INFO_PATH], ['train', 'train_info'], ['numpy', 'pickle']
        )
        print('done.')
        print('recovering test (sequence, target) pairs...', end=' ', flush=True)
        data_splitter = get_splitter_classes()[args.modeling_split](args)
        if task_type == 'forecasting':
            kwargs = {'n_back': args.n_back, 'n_forward': args.n_forward}
        else:
            kwargs = {'window_size': args.window_size, 'window_step': args.window_step}
        data = data_splitter.get_modeling_split(modeling_files['train'], modeling_files['train_info'], **kwargs)
        print('done.')
        detector_fit_args = {k.replace('_test', ''): v for k, v in data.items() if 'test' in k}
        looped_arg_names += ['thresholding_factor', 'n_iterations', 'removal_factor']
        values_combinations = itertools.product(*[args_dict[looped] for looped in looped_arg_names])
        # remove variations of `removal_factor` if `n_iterations` is 1, as it does not apply
        values_combinations = set([tuple([*v[:-1], 1.0]) if v[2] == 1 else tuple(v) for v in values_combinations])

    # supervised threshold selection
    else:
        detector_fit_args = {
            'periods': thresholding_data['threshold'],
            'periods_labels': thresholding_data['y_threshold'],
            'evaluator': evaluator
        }
        looped_arg_names += ['n_bins', 'n_bin_trials']
        values_combinations = itertools.product(*[args_dict[looped] for looped in looped_arg_names])

    # perform and evaluate final predictions for every thresholding combination
    for arg_values in values_combinations:
        # update command-line arguments and paths
        for i, arg_value in enumerate(arg_values):
            args_dict[looped_arg_names[i]] = arg_value
        args = argparse.Namespace(**args_dict)
        OUTPUT_PATH = get_output_path(args, 'train_detector', 'model')
        COMPARISON_PATH = get_output_path(args, 'train_detector', 'comparison')

        # build the final anomaly detector using the scorer and command-line arguments
        detector = Detector(args, scorer, OUTPUT_PATH)

        # select the threshold maximizing the performance on `threshold`
        detector.fit(**detector_fit_args)

        # derive the detector's record-wise predictions for the disturbed periods
        thresholding_processed = {k: v for k, v in thresholding_data.items() if k not in thresholding_sets}
        for set_name in thresholding_sets:
            thresholding_processed[f'{set_name}_preds'] = detector.predict(thresholding_data[set_name])

        # save global and type-wise performance if the data contains multiple anomaly types
        config_name = hyper_to_path(args.threshold_selection, *get_thresholding_args(args))
        save_detection_evaluation(
            thresholding_processed, evaluator, get_evaluation_string(args), config_name, COMPARISON_PATH,
            event_types=event_types, metrics_colors=metrics_colors, method_path=OUTPUT_PATH
        )
