"""Spark-specific utility module.
"""
import os
import copy
import argparse
from itertools import permutations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATA_ROOT = os.getenv('DATA_ROOT')

# spark streaming trace types
TRACE_TYPES = [
    'bursty_input', 'bursty_input_crash', 'stalled_input',
    'cpu_contention', 'process_failure'
]
# event types (the integer label for a type is its corresponding index in the list +1)
EVENT_TYPES = [
    'bursty_input', 'bursty_input_crash', 'stalled_input',
    'cpu_contention', 'driver_failure', 'executor_failure', 'unknown'
]

# path extensions from the root path
PATH_EXTENSIONS = {'undisturbed': 'undisturbed'}
for type_ in TRACE_TYPES:
    PATH_EXTENSIONS[type_] = os.path.join('disturbed', type_)

# full data paths for the Spark traces and anomaly spreadsheet
DATA_PATHS_DICT = {'labels': DATA_ROOT}
for k, extension in PATH_EXTENSIONS.items():
    DATA_PATHS_DICT[k] = os.path.join(DATA_ROOT, extension)

# raw sampling period of records
SAMPLING_PERIOD = '1s'

# spark-specific default values for the common command-line arguments
DEFAULT_ARGS = {
    # datasets constitution arguments
    'data': 'spark',
    'threshold_supervision': 'unsupervised',
    'primary_start': 'na',
    'secondary_start': 'na',
    'secondary_end': 'na',
    'tolerance_end': 'na',
    'pre_sampling_period': '15s',

    # features alteration and transformation arguments
    'alter_bundles': 'spark_bundles',
    'alter_bundle_idx': 1,
    'sampling_period': '15s',
    'transform_chain': 'trace_scaling',
    # if a transformation step is repeated, the same arguments are used for all its instances
    'head_size': 240,
    'online_window_type': 'expanding',
    # if not -1, weight of a regular pretraining of the standard scaler in
    # the convex combination with its head/head-online training
    'regular_pretraining_weight': -1,
    'scaling_method': 'std',
    'minmax_range': [0, 1],
    'n_components': 19,

    # normality modeling arguments
    'modeling_split': 'stratified.split',
    'n_period_strata': 3,
    'modeling_val_prop': 0.15,
    'modeling_test_prop': 0.15,
    'model_type': 'ae',
    # FORECASTING MODELS #
    'n_back': 40,
    'n_forward': 1,
    # RNN
    'rnn_unit_type': 'lstm',
    'rnn_n_hidden_neurons': [50],
    'rnn_dropout': 0.0,
    'rnn_rec_dropout': 0.0,
    'rnn_optimizer': 'adam',
    'rnn_learning_rate': 2 * (10 ** -4),
    'rnn_n_epochs': 200,
    'rnn_batch_size': 32,
    # RECONSTRUCTION MODELS #
    'window_size': 40,
    'window_step': 1,
    # simple autoencoder
    'ae_latent_dim': 32,
    'ae_type': 'dense',
    'ae_enc_n_hidden_neurons': [50],
    'ae_dec_last_activation': 'linear',
    'ae_dropout': 0.0,
    'ae_rec_unit_type': 'lstm',
    'ae_rec_dropout': 0.0,
    'ae_loss': 'mse',
    'ae_optimizer': 'adam',
    'ae_learning_rate': 2 * (10 ** -4),
    'ae_n_epochs': 200,
    'ae_batch_size': 32,
    # BiGAN
    'bigan_latent_dim': 32,
    'bigan_enc_type': 'rec',
    'bigan_enc_arch_idx': -1,
    'bigan_enc_rec_n_hidden_neurons': [100],
    'bigan_enc_rec_unit_type': 'lstm',
    'bigan_enc_conv_n_filters': 32,
    'bigan_enc_dropout': 0.0,
    'bigan_enc_rec_dropout': 0.0,
    'bigan_gen_type': 'rec',
    'bigan_gen_last_activation': 'linear',
    'bigan_gen_arch_idx': -1,
    'bigan_gen_rec_n_hidden_neurons': [100],
    'bigan_gen_rec_unit_type': 'lstm',
    'bigan_gen_conv_n_filters': 64,
    'bigan_gen_dropout': 0.0,
    'bigan_gen_rec_dropout': 0.0,
    'bigan_dis_type': 'conv',
    'bigan_dis_arch_idx': 0,
    'bigan_dis_x_rec_n_hidden_neurons': [30, 10],
    'bigan_dis_x_rec_unit_type': 'lstm',
    'bigan_dis_x_conv_n_filters': 32,
    'bigan_dis_x_dropout': 0.0,
    'bigan_dis_x_rec_dropout': 0.0,
    'bigan_dis_z_n_hidden_neurons': [32, 10],
    'bigan_dis_z_dropout': 0.0,
    'bigan_dis_threshold': 0.0,
    'bigan_dis_optimizer': 'adam',
    'bigan_enc_gen_optimizer': 'adam',
    'bigan_dis_learning_rate': 0.0004,
    'bigan_enc_gen_learning_rate': 0.0001,
    'bigan_n_epochs': 200,
    'bigan_batch_size': 32,

    # outlier score assignment arguments
    'scoring_method': 'mse',
    'mse_weight': 0.5,

    # supervised evaluation for assessing scoring performance
    'evaluation_type': 'regular',
    'beta': 1.0,
    'recall_alpha': 0.0,
    'recall_omega': 'default',
    'recall_delta': 'flat',
    'recall_gamma': 'dup',
    'precision_omega': 'default',
    'precision_delta': 'flat',
    'precision_gamma': 'dup',

    # outlier score threshold calibration arguments
    'threshold_selection': ['std', 'mad', 'iqr'],
    # supervised threshold selection arguments
    'supervised_threshold_target': 'avg.perf',
    'n_bins': [10],
    'n_bin_trials': [50],
    # unsupervised threshold selection arguments
    'thresholding_factor': [1.5, 2.0, 2.5, 3.0],
    'n_iterations': [1, 2],
    'removal_factor': [1.0],

    # explanation discovery arguments
    'explanation_method': 'exstream',
    'explained_predictions': 'ground.truth',
    # explanation evaluation parameters
    'exp_eval_min_sample_length': 45,
    'exp_eval_min_anomaly_length': 5,
    'exp_eval_n_runs': 5,
    'exp_eval_test_prop': 0.2,
    # EXstream hyperparameters
    'exstream_tolerance': 0.9,
    'exstream_correlation_threshold': 0.5,
    # MacroBase hyperparameters
    'macrobase_min_risk_ratio': 1.0,
    'macrobase_n_bins': 10,
    'macrobase_min_support': 0.4,
    # 'macrobase_min_support': 0.5,
    # LIME hyperparameters
    'lime_n_features': 5,

    # full pipeline running arguments
    'pipeline_type': 'ad'
}


# returns the application number of a file based on its name
def file_app(file_name): return int(file_name.split('_')[2])


def is_type_combination(x):
    """Argparse parsing function: returns `x` if it is either `.` or any combination of trace types.

    Note: we use this instead of the `choices` parameter because the amount of choices otherwise
    floods the argument's `help`.

    Args:
        x (str): the command line argument to be checked.

    Returns:
        str: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    if x == '.':
        return x
    trace_types = list(PATH_EXTENSIONS.keys())
    type_choices = ['.'.join(t) for r in range(1, len(trace_types) + 1) for t in permutations(trace_types, r)]
    if x in type_choices:
        return x
    raise argparse.ArgumentTypeError('Argument has to be either `.` for all trace types, '
                                     f'or else any dot-separated combination of {trace_types}')


def add_specific_args(parsers, pipeline_step, pipeline_steps):
    """Adds Spark-specific command-line arguments to the input parsers.

    The arguments added to the provided step are propagated to the next ones in the pipeline.

    Args:
        parsers (dict): the existing parsers dictionary.
        pipeline_step (str): the pipeline step to add arguments to (by the name of its main script).
        pipeline_steps (list): the complete list of step names in the main AD pipeline.

    Returns:
        dict: the new parsers extended with Spark-specific arguments for the step.
    """
    step_index = pipeline_steps.index(pipeline_step)
    new_parsers = copy.deepcopy(parsers)
    arg_names, arg_params = [], []
    if pipeline_step == 'make_datasets':
        # considered traces might be restricted to a given application id (0 means no app restriction)
        app_ids = [1, 2, 3, 4, 7, 8, 12, 14]
        app_choices = [0] + app_ids
        arg_names.append('--app-id')
        arg_params.append({
            'default': 14,
            'choices': app_choices,
            'type': int,
            'help': 'application id (0 if all)'
        })
        # considered traces might be restricted to a set of trace types ('.' means no type restriction)
        arg_names.append('--trace-types')
        arg_params.append({
            'default': '.',
            'type': is_type_combination,
            'help': 'restricted trace types to consider (dot-separated, `.` for no restriction)'
        })
    for i, arg_name in enumerate(arg_names):
        for step in pipeline_steps[step_index:]:
            new_parsers[step].add_argument(arg_name, **arg_params[i])
    return new_parsers
