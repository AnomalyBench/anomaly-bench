"""Explanation discovery module.

The "Explainer" objects we currently use do not actually require any training. They
operate on `[normal + anomaly]` sequences we call explanation "samples".
"""
import os
import time
import argparse
import importlib

import numpy as np
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    hyper_to_path, parsers, get_output_path, get_dataset_names, model_based_explanation_choices,
    forecasting_choices, reconstruction_choices, get_explanation_args, get_evaluation_string
)
from utils.spark import EVENT_TYPES
from data.helpers import load_datasets_data, load_mixed_formats
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.forecasters import forecasting_classes
from modeling.reconstruction.reconstructors import reconstruction_classes
from detection.detector import Detector
from explanation.explainers import get_explanation_classes
from explanation.evaluation import save_explanation_evaluation


def get_best_thresholding_args(args):
    thresholding_comparison_path = get_output_path(args, 'train_detector', 'comparison')
    full_thresholding_comparison_path = os.path.join(
        thresholding_comparison_path, f'{get_evaluation_string(args)}_detection_comparison.csv'
    )
    thresholding_comparison_df = pd.read_csv(full_thresholding_comparison_path, index_col=[0, 1]).astype(float)
    best_thresholding_row = thresholding_comparison_df.loc[
        thresholding_comparison_df.index.get_level_values('granularity') == 'app_avg'
        ].sort_values('TEST_GLOBAL_F1.0_SCORE', ascending=False).iloc[0]
    return get_new_thresholding_args_from_str(args, best_thresholding_row.name[0])


def get_new_thresholding_args_from_str(args, model_str):
    # split and remove training timestamp
    model_args = model_str.split('_')
    # update and return args
    args_dict = vars(args)
    args_dict['threshold_selection'] = model_args[0]
    args_dict['thresholding_factor'] = float(model_args[1])
    args_dict['n_iterations'] = int(model_args[2])
    args_dict['removal_factor'] = float(model_args[3])
    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['train_explainer'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')

    # load the periods records, labels and information used to train and evaluate explanation discovery
    explanation_sets = get_dataset_names(args.threshold_supervision, disturbed_only=True)
    explanation_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, explanation_sets)

    # load AD model and/or model predictions if relevant
    explained_model, normal_model_samples = None, None
    if args.explanation_method in model_based_explanation_choices or \
            args.explained_predictions == 'model.predictions':
        # modeling and scoring output paths
        MODEL_INPUT_PATH = get_output_path(args, 'train_model')
        SCORER_INPUT_PATH = get_output_path(args, 'train_scorer')
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

        if args.explanation_method in model_based_explanation_choices:
            explained_model = scorer
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
            if 'y_test' not in data:
                normal_model_samples = data['X_test']
            else:
                normal_model_samples = np.array([(x, y) for x, y in zip(data['X_test'], data['y_test'])])

        if args.explained_predictions == 'model.predictions':
            # if not a single set of thresholding parameters was passed, pick the best tried for the method
            if type(args.threshold_selection) == list:
                args = get_best_thresholding_args(args)
            # load the final detector with relevant threshold and replace labels with model predictions
            THRESHOLD_PATH = get_output_path(args, 'train_detector')
            detector = Detector.from_file(args, scorer, os.path.join(THRESHOLD_PATH, 'threshold.pkl'))
            for set_name in explanation_sets:
                explanation_data[f'y_{set_name}'] = detector.predict(explanation_data[set_name])
                # prepend negative predictions for the first records used by the model if forecasting-based
                if args.model_type in forecasting_choices:
                    for i in range(len(explanation_data[f'y_{set_name}'])):
                        explanation_data[f'y_{set_name}'][i] = np.concatenate([
                            np.zeros(shape=(args.n_back,)), explanation_data[f'y_{set_name}'][i]
                        ])
    # set output and comparison paths here to use potentially updated args
    OUTPUT_PATH = get_output_path(args, 'train_explainer', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_explainer', 'comparison')
    # initialize the relevant explainer based on command-line arguments
    explainer = get_explanation_classes()[args.explanation_method](
        args, OUTPUT_PATH, explained_model, normal_model_samples
    )
    # evaluate the explanations on the explanation data
    config_name = hyper_to_path(
        args.explanation_method, *get_explanation_args(args), time.strftime('run.%Y.%m.%d.%H.%M.%S')
    )
    event_types = EVENT_TYPES if args.data == 'spark' else None
    save_explanation_evaluation(
        args, explanation_data, explainer, config_name, COMPARISON_PATH, event_types=event_types
    )
