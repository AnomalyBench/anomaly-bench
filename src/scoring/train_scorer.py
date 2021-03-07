"""Outlier score assignment module.

From the provided model, trains an outlier score derivation method and evaluates it
on the labeled dataset(s).
"""
import os
import importlib

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    hyper_to_path, parsers, get_output_path, get_dataset_names, get_scoring_args,
    get_evaluation_string, forecasting_choices, reconstruction_choices
)
from data.helpers import load_datasets_data, load_mixed_formats

from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from modeling.forecasting.forecasters import forecasting_classes
from modeling.reconstruction.reconstructors import reconstruction_classes

from scoring.evaluation import save_scoring_evaluation

from visualization.helpers.spark import EVENT_TYPES
from metrics.evaluators import evaluation_classes


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['train_scorer'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')
    MODEL_INPUT_PATH = get_output_path(args, 'train_model')
    OUTPUT_PATH = get_output_path(args, 'train_scorer', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_scorer', 'comparison')

    # load the periods records, labels and information used to evaluate the outlier scores
    scoring_sets = get_dataset_names(args.threshold_supervision, disturbed_only=True)
    scoring_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, scoring_sets)

    # load the model and scoring classes for the relevant type of task
    a_m = 'supported models are limited to forecasting-based and reconstruction-based models'
    assert args.model_type in forecasting_choices + reconstruction_choices, a_m
    task_type = 'forecasting' if args.model_type in forecasting_choices else 'reconstruction'
    model_classes = forecasting_classes if task_type == 'forecasting' else reconstruction_classes
    scoring_classes = importlib.import_module(f'scoring.{task_type}.scorers').scoring_classes

    if task_type == 'forecasting':
        # adapt labels to the task cutting out the first `n_back` records of each period
        for n in scoring_sets:
            scoring_data[f'y_{n}'] = get_trimmed_periods(scoring_data[f'y_{n}'], args.n_back)

    # non-parametric models are simply initialized without loading anything
    if args.model_type == 'naive.forecasting':
        model = model_classes[args.model_type](args, '')
    # others are loaded from disk
    else:
        full_model_path = os.path.join(MODEL_INPUT_PATH, 'model.h5')
        model = model_classes[args.model_type].from_file(args, full_model_path)

    # initialize the relevant scorer based on command-line arguments
    scorer = scoring_classes[args.scoring_method](args, model, OUTPUT_PATH)
    # parametric scoring methods are fit on the same validation set as the model
    if args.scoring_method == 'nll':
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
        X_test, y_test = data['X_test'], data['y_test']
        print('done.')
        scorer.fit(X_test, y_test)
    # others are simply used without any fitting procedure
    for set_name in scoring_sets:
        scoring_data[f'{set_name}_scores'] = scorer.score(scoring_data[set_name])
        scoring_data.pop(set_name)

    # save the scoring evaluation on the labeled threshold and test sets for each event type
    event_types = EVENT_TYPES if args.data == 'spark' else None
    metrics_colors = importlib.import_module(f'visualization.helpers.{args.data}').METRICS_COLORS
    config_name = hyper_to_path(args.scoring_method, *get_scoring_args(args))
    evaluator = evaluation_classes[args.evaluation_type](args)
    save_scoring_evaluation(
        scoring_data, evaluator, get_evaluation_string(args), config_name, COMPARISON_PATH,
        event_types=event_types, metrics_colors=metrics_colors, method_path=OUTPUT_PATH
    )
