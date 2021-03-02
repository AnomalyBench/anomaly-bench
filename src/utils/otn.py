"""OTN-specific utility module.

TODO - update arguments to support unsupervised threshold selection.
"""
import os
import copy

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATA_ROOT = os.getenv('DATA_ROOT')

# full data paths for the recorded services and labels
DATA_PATHS_DICT = dict()
for key, folder in zip(['traces', 'labels'], ['TimeSeries_no_missing_value', 'W_predictable']):
    DATA_PATHS_DICT[key] = os.path.join(DATA_ROOT, folder)

# raw sampling period of records
SAMPLING_PERIOD = '15m'


# otn-specific default values for the common command-line arguments
DEFAULT_ARGS = {
    # datasets constitution arguments
    'data': 'otn',
    'primary_start': '-2d',
    'secondary_start': '-1d',
    'secondary_end': '0',
    'tolerance_end': '1d',
    'pre_sampling_period': '2h',

    # features alteration and transformation arguments
    'alter_bundles': 'otn_bundles',
    'alter_bundle_idx': 0,
    'sampling_period': '2h',
    'transform_chain': 'regular_scaling',
    # if a transformation step is repeated, the same arguments are used for all its instances
    'head_size': 672,
    'online_window_type': 'expanding',
    'scaling_method': 'minmax',
    'minmax_range': [0, 1],
    'explained_variance': 0.99,

    # normality modeling arguments
    'modeling_split': 'service.split',
    'n_period_strata': 3,
    'model_type': 'ae',
    # FORECASTING MODELS #
    'n_back': 36,
    'n_forward': 1,
    # RNN
    'rnn_unit_type': 'lstm',
    'rnn_n_hidden_neurons': [150],
    'rnn_dropout': 0.0,
    'rnn_rec_dropout': 0.0,
    'rnn_optimizer': 'nadam',
    'rnn_learning_rate': 2.026 * (10 ** -4),
    'rnn_n_epochs': 200,
    'rnn_batch_size': 32,
    # RECONSTRUCTION MODELS #
    'window_size': 36,
    'window_step': 1,
    # simple autoencoder
    'ae_latent_dim': 15,
    'ae_type': 'rec',
    'ae_enc_n_hidden_neurons': [30],
    'ae_dropout': 0.0,
    'ae_rec_unit_type': 'lstm',
    'ae_rec_dropout': 0.0,
    'ae_optimizer': 'adam',
    'ae_learning_rate': 2 * (10 ** -4),
    'ae_n_epochs': 200,
    'ae_batch_size': 32,
    # BiGAN
    'bigan_latent_dim': 15,
    'bigan_enc_type': 'rec',
    'bigan_enc_arch_idx': -1,
    'bigan_enc_rec_n_hidden_neurons': [30],
    'bigan_enc_rec_unit_type': 'lstm',
    'bigan_enc_conv_n_filters': 32,
    'bigan_enc_dropout': 0.0,
    'bigan_enc_rec_dropout': 0.0,
    'bigan_gen_type': 'rec',
    'bigan_gen_arch_idx': -1,
    'bigan_gen_rec_n_hidden_neurons': [30],
    'bigan_gen_rec_unit_type': 'lstm',
    'bigan_gen_conv_n_filters': 64,
    'bigan_gen_dropout': 0.0,
    'bigan_gen_rec_dropout': 0.0,
    'bigan_dis_type': 'conv',
    'bigan_dis_arch_idx': -1,
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
    'mse_weight': 0.9,

    # supervised evaluation for assessing scoring performance
    'evaluation_type': 'fast.alert',
    'beta': 0.5,
    'recall_alpha': 0.0,
    'recall_delta': 'flat',
    'recall_gamma': 'dup',
    'precision_delta': 'flat',
    'precision_gamma': 'dup',

    # outlier score threshold calibration arguments
    'threshold_optimization': 'search',
    'threshold_target': 'global.perf',
    'n_bins': 10,
    'n_bin_trials': 50
}


def add_specific_args(parsers, pipeline_step, pipeline_steps):
    """Adds OTN-specific command-line arguments to the input parsers.

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
        # optional considered services limit (keeping the normal/anomalous distribution)
        arg_names.append('--n-services')
        arg_params.append({
            'default': 200,
            'type': int,
            'help': 'number of service traces to consider (-1 for all)'
        })
    for i, arg_name in enumerate(arg_names):
        for step in pipeline_steps[step_index:]:
            new_parsers[step].add_argument(arg_name, **arg_params[i])
    return new_parsers
