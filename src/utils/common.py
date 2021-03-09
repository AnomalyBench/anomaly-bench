"""General utility module. Mainly handling paths and command line arguments.
"""
import os
import argparse
import importlib

import tensorflow as tf
from dotenv import load_dotenv, find_dotenv

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)

# tensorflow version used throughout the project
TF_VERSION = 1 if tf.__version__[:2] == '1.' else 2

# main AD pipeline step names
PIPELINE_STEPS = [
    'make_datasets',
    'build_features',
    'train_model',
    'train_scorer',
    'train_detector',
    'train_explainer'
]

# output paths
load_dotenv(find_dotenv())
OUTPUTS_ROOT = os.getenv('OUTPUTS_ROOT')
INTERIM_ROOT = os.path.join(OUTPUTS_ROOT, 'data', 'interim')
PROCESSED_ROOT = os.path.join(OUTPUTS_ROOT, 'data', 'processed')
MODELS_ROOT = os.path.join(OUTPUTS_ROOT, 'models')

# a lot of the default command-line arguments depend on the data we use
DEFAULTS = importlib.import_module(f'utils.{os.getenv("USED_DATA").lower()}').DEFAULT_ARGS


def hyper_to_path(*parameters):
    """Returns the path extension corresponding to the provided parameter list.
    """
    # empty string parameters are ignored (and `[a, b]` lists are turned to `[a-b]`)
    parameters = [str(p).replace(', ', '-') for p in parameters if p != '']
    return '_'.join(map(str, parameters))


def get_output_path(args, pipeline_step, output_details=None):
    """Returns the output path for the specified step of the AD pipeline according to the command line arguments.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        pipeline_step (str): step of the AD pipeline, as the name of the main python file (without extension).
        output_details (str|None): additional details in case there are several possible output paths for the step.

    Returns:
        str: the full output path.
    """
    path_extensions = dict()
    step_index = PIPELINE_STEPS.index(pipeline_step)
    short_args = get_short_args(args, step_index)
    if args.data == 'spark':
        data_args = [args.app_id]
        if args.trace_types != '.':
            data_args.append(short_args.trace_types)
    else:
        data_args = []
        if args.n_services != -1:
            data_args += [args.n_services]
        data_args += [args.primary_start, args.secondary_start, args.secondary_end, args.tolerance_end]
    path_extensions['make_datasets'] = os.path.join(
        hyper_to_path(
            args.data,
            *data_args,
            short_args.threshold_supervision,
            args.pre_sampling_period
        )
    )
    if step_index >= 1:
        alter_args = []
        if args.alter_bundles != '.':
            # remove the word `bundles` to shorten the path
            alter_args += [short_args.alter_bundles, args.alter_bundle_idx]
        transform_args = [short_args.transform_chain.replace('_', '-')] if args.transform_chain != '.' else ['']
        # if the same transform step is repeated, the same arguments are used for all instances
        if 'head' in args.transform_chain:
            head_args = [args.head_size, args.regular_pretraining_weight]
            if 'online' in args.transform_chain:
                head_args = [short_args.online_window_type] + head_args
            transform_args += head_args
        if 'scaling' in args.transform_chain:
            transform_args.append(args.scaling_method)
            if args.scaling_method == 'minmax':
                transform_args.append(args.minmax_range)
        if 'pca' in args.transform_chain:
            transform_args.append(args.n_components)
        path_extensions['build_features'] = os.path.join(
            hyper_to_path(args.sampling_period, *alter_args, *transform_args)
        )
    if step_index >= 2:
        # separate arguments of the modeling task from the ones of the model performing that task
        splitting_args = [short_args.modeling_split]
        if args.modeling_split == 'stratified.split':
            splitting_args += [args.n_period_strata]
        splitting_args += [args.modeling_val_prop, args.modeling_test_prop]
        task_name, task_args = '', []
        if args.model_type in forecasting_choices:
            task_name = 'fore'
            task_args += [args.n_back, args.n_forward]
        elif args.model_type in reconstruction_choices:
            task_name = 'reco'
            task_args += [args.window_size, args.window_step]
        model_args = get_model_args(args)
        path_extensions['train_model'] = os.path.join(
            hyper_to_path('ad', *splitting_args, task_name, *task_args),
            hyper_to_path(args.model_type, *model_args)
        )
    if step_index >= 3:
        scoring_args = [args.scoring_method]
        if args.scoring_method in ['mse.dis', 'mse.ft']:
            scoring_args += [args.mse_weight]
        path_extensions['train_scorer'] = os.path.join(
            hyper_to_path(*scoring_args)
        )
    if step_index >= 4:
        # threshold selection and AD evaluation
        thresholding_args = []
        if args.threshold_selection == 'search':
            thresholding_args += [args.n_bins, args.n_bin_trials]
        if args.threshold_selection in two_stat_ts_sel_choices:
            thresholding_args += [args.thresholding_factor, args.n_iterations, args.removal_factor]
        # output path depends on the threshold selection supervision
        evaluation_args = [get_evaluation_string(args)] if args.threshold_supervision == 'supervised' else []
        path_extensions['train_detector'] = os.path.join(
            *evaluation_args, hyper_to_path(args.threshold_selection, *thresholding_args)
        )
    if step_index >= 5:
        # explanation discovery
        explanation_args = get_explanation_args(args)
        path_extensions['train_explainer'] = os.path.join(
            hyper_to_path(args.explanation_method, *explanation_args)
        )
    if pipeline_step == 'make_datasets':
        return os.path.join(
            INTERIM_ROOT,
            path_extensions['make_datasets']
        )
    if pipeline_step == 'build_features':
        # we consider we want the output *data* path by default
        return os.path.join(
            PROCESSED_ROOT if output_details is None or output_details == 'data' else MODELS_ROOT,
            path_extensions['make_datasets'],
            path_extensions['build_features']
        )
    # either `train_model`, `train_scorer`, `train_detector` or `train_explainer` here
    if pipeline_step == 'train_explainer' and args.explained_predictions == 'ground.truth':
        # no AD model involved in the explanations at all
        if args.explanation_method in model_free_explanation_choices:
            # the chain is then just data => features => explanation
            chain_ids = [PIPELINE_STEPS.index('make_datasets'), PIPELINE_STEPS.index('build_features'), step_index]
        # no model predictions involved
        else:
            # the chain then stops at `train_scorer`, the explained model being for now a scorer only
            chain_ids = list(range(PIPELINE_STEPS.index('train_scorer') + 1)) + [step_index]
        extensions_chain = [path_extensions[PIPELINE_STEPS[i]] for i in chain_ids]
        # add "ed" prefix to visually separate from anomaly detection tasks
        extensions_chain[-1] = hyper_to_path('ed', extensions_chain[-1])
    else:
        extensions_chain = [path_extensions[PIPELINE_STEPS[i]] for i in range(step_index + 1)]
    # return either the current step model's path (default) or the step comparison path
    comparison_path = os.path.join(
        MODELS_ROOT,
        *extensions_chain[:-1]
    )
    if output_details is None or output_details == 'model':
        return os.path.join(comparison_path, extensions_chain[-1])
    assert output_details == 'comparison', f'specify `comparison` for the comparison path of `{pipeline_step}`'
    return comparison_path


def get_short_args(args, step_index):
    """Returns shorten values for the command-line arguments to be used in paths.

    We might indeed encounter some issues when using long paths with various libraries.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        step_index (int): index of the AD pipeline step (starting at 0).

    Returns:
        argparse.Namespace: the same arguments, with shorten values to be used in paths.
    """
    short_args = argparse.Namespace(**vars(args))
    if args.data == 'spark':
        short_args.trace_types = args.trace_types.replace('_contention', '').replace('_input', '')
    short_args.threshold_supervision = args.threshold_supervision[:3] + '.ts'
    if step_index >= 1:
        if args.alter_bundles != '.':
            short_args.alter_bundles = args.alter_bundles.replace('_bundles', '')
        short_args.transform_chain = args.transform_chain.replace(
            'scaling', 'scl'
        ).replace('head', 'hd').replace('online', 'oln').replace('regular', 'reg')
        short_args.online_window_type = args.online_window_type.replace('expanding', 'exp').replace('rolling', 'rol')
    if step_index >= 2:
        short_args.modeling_split = args.modeling_split.replace('.split', '')
    if step_index >= 3:
        pass
    if step_index >= 4:
        pass
    return short_args


def get_dataset_names(threshold_supervision, disturbed_only=False):
    """Returns the dataset names relevant to the threshold selection strategy.

    - Supervised selection: a labeled `threshold` dataset is used to select the outlier
        score threshold (mixed labeled data).
    - Unsupervised selection: a part of `train` (modeling test set) is used to select
        the outlier score threshold (unlabeled, assumed normal, data).

    Args:
        threshold_supervision (str): threshold selection strategy.
            Either `supervised` or `unsupervised`.
        disturbed_only (bool): whether to return dataset names with disturbed traces only.
            i.e. whether to drop `train`.

    Returns:
        list: list of dataset names corresponding to the strategy and `disturbed_only`.
    """
    a_t = 'threshold selection supervision must be either `supervised` or `unsupervised`'
    assert threshold_supervision in threshold_sup_choices, a_t
    if threshold_supervision == 'supervised':
        set_names = ['train', 'threshold', 'test']
    else:
        set_names = ['train', 'test']
    if disturbed_only:
        return set_names[1:]
    return set_names


def get_modeling_task_string(args):
    """Returns the arguments string of the modeling task based on the command-line arguments.

    This string can be used as a prefix for models comparison to make sure they are compared
    for the same target task.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        str: the task string, in the regular path-like format.
    """
    full_task_args = [args.modeling_split]
    if args.modeling_split == 'stratified.split':
        full_task_args += [args.n_period_strata]
    full_task_args += [args.modeling_val_prop, args.modeling_test_prop]
    if args.model_type in forecasting_choices:
        full_task_args += [args.n_back, args.n_forward, 'forecasting']
    elif args.model_type in reconstruction_choices:
        full_task_args += [args.window_size, args.window_step, 'reconstruction']
    # add "ad" prefix to visually separate from explanation discovery tasks
    return hyper_to_path('ad', *full_task_args)


def get_model_args(args):
    """Returns a relevant and ordered list of model argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of model argument values corresponding to `args`.
    """
    model_args = []
    if args.model_type == 'rnn':
        model_args = [
            args.rnn_unit_type, args.rnn_n_hidden_neurons,
            f'{args.rnn_dropout:.2f}', f'{args.rnn_rec_dropout:.2f}',
            args.rnn_optimizer, f'{args.rnn_learning_rate:.7f}',
            args.rnn_n_epochs, args.rnn_batch_size
        ]
    if args.model_type == 'ae':
        rec_unit_type, rec_dropout = [], []
        if args.ae_type == 'rec':
            rec_unit_type, rec_dropout = [args.ae_rec_unit_type], [args.ae_rec_dropout]
        model_args = [
            args.ae_type, *rec_unit_type,
            args.ae_enc_n_hidden_neurons, args.ae_dec_last_activation,
            args.ae_latent_dim, args.ae_dropout, *rec_dropout,
            args.ae_loss, args.ae_optimizer, f'{args.ae_learning_rate:.7f}',
            args.ae_n_epochs, args.ae_batch_size
        ]
    if args.model_type == 'bigan':
        enc_args, gen_args = [args.bigan_enc_type], [args.bigan_gen_type, args.bigan_gen_last_activation]
        dis_args = [args.bigan_dis_type]
        # encoder network arguments
        if args.bigan_enc_arch_idx != -1:
            enc_args.append(args.bigan_enc_arch_idx)
        else:
            enc_dropout_args = [args.bigan_enc_dropout]
            if args.bigan_enc_type == 'rec':
                enc_args += [
                    args.bigan_enc_rec_n_hidden_neurons,
                    args.bigan_enc_rec_unit_type,
                ]
                enc_dropout_args.append(args.bigan_enc_rec_dropout)
            elif args.bigan_enc_type == 'conv':
                enc_args += [
                    args.bigan_enc_conv_n_filters
                ]
            enc_args += enc_dropout_args
        # generator network arguments
        if args.bigan_gen_arch_idx != -1:
            gen_args.append(args.bigan_gen_arch_idx)
        else:
            gen_dropout_args = [args.bigan_gen_dropout]
            if args.bigan_gen_type == 'rec':
                gen_args += [
                    args.bigan_gen_rec_n_hidden_neurons,
                    args.bigan_gen_rec_unit_type,
                ]
                gen_dropout_args.append(args.bigan_gen_rec_dropout)
            elif args.bigan_gen_type == 'conv':
                gen_args += [
                    args.bigan_gen_conv_n_filters
                ]
            gen_args += gen_dropout_args
        # discriminator network arguments
        if args.bigan_dis_arch_idx != -1:
            dis_args.append(args.bigan_dis_arch_idx)
        else:
            # x path
            dis_dropout_args = [args.bigan_dis_x_dropout]
            if args.bigan_dis_type == 'rec':
                dis_args += [
                    args.bigan_dis_x_rec_n_hidden_neurons,
                    args.bigan_dis_x_rec_unit_type,
                ]
                dis_dropout_args.append(args.bigan_dis_x_rec_dropout)
            elif args.bigan_dis_type == 'conv':
                dis_args += [
                    args.bigan_dis_x_conv_n_filters
                ]
            dis_args += dis_dropout_args
            # z path
            dis_args += [args.bigan_dis_z_n_hidden_neurons, args.bigan_dis_z_dropout]
        model_args = [
            args.bigan_latent_dim,
            *enc_args, *gen_args, *dis_args,
            f'{args.bigan_dis_threshold:.2f}',
            args.bigan_dis_optimizer, f'{args.bigan_dis_learning_rate:.7f}',
            args.bigan_enc_gen_optimizer, f'{args.bigan_enc_gen_learning_rate:.7f}',
            args.bigan_n_epochs, args.bigan_batch_size
        ]
    return model_args


def get_scoring_args(args):
    """Returns a relevant and ordered list of scoring argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of scoring argument values corresponding to `args`.
    """
    scoring_args = []
    if args.scoring_method in ['mse.dis', 'mse.ft']:
        scoring_args.append(args.mse_weight)
    return scoring_args


def get_explanation_args(args):
    """Returns a relevant and ordered list of explanation discovery argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of explanation discovery argument values corresponding to `args`.
    """
    explanation_args = []
    if args.explanation_method == 'exstream':
        explanation_args += [args.exstream_tolerance, args.exstream_correlation_threshold]
    if args.explanation_method == 'macrobase':
        explanation_args += [args.macrobase_min_risk_ratio, args.macrobase_n_bins, args.macrobase_min_support]
    if args.explanation_method == 'lime':
        explanation_args += [args.lime_n_features]
    return explanation_args


def get_thresholding_args(args):
    """Returns a relevant and ordered list of thresholding argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of thresholding argument values corresponding to `args`.
    """
    thresholding_args = []
    if args.threshold_selection == 'search':
        thresholding_args += [args.n_bins, args.n_bin_trials]
    if args.threshold_selection in two_stat_ts_sel_choices:
        thresholding_args += [args.thresholding_factor, args.n_iterations, args.removal_factor]
    return thresholding_args


def get_evaluation_string(args):
    """Returns the evaluation arguments string based on the command-line arguments.

    This string can be used as a prefix for models comparison to make sure they are compared
    under the same requirements.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        str: the evaluation string, in the regular path-like format.
    """
    full_evaluation_args = [args.evaluation_type, args.beta]
    if args.evaluation_type == 'range':
        full_evaluation_args += [
            args.recall_alpha, args.recall_omega, args.recall_delta, args.recall_gamma,
            args.precision_omega, args.precision_delta, args.precision_gamma
        ]
    return hyper_to_path(*full_evaluation_args)


def get_explanation_evaluation_str(args):
    """Returns the explanation discovery evaluation arguments string based on the command-line arguments.

    This string can be used as a prefix for models comparison to make sure they are compared
    under the same requirements.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        str: the evaluation string, in the regular path-like format.
    """
    exp_evaluation_args = [
        args.exp_eval_min_sample_length, args.exp_eval_min_anomaly_length,
        args.exp_eval_n_runs, args.exp_eval_test_prop
    ]
    return hyper_to_path('ed', *exp_evaluation_args)


def is_percentage(x):
    """Argparse parsing function: returns `x` as a float if it is between 0 and 1.

    Args:
        x (str): the command line argument to be checked.

    Returns:
        float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if not (0 <= x <= 1):
        raise argparse.ArgumentTypeError('Argument has to be between 0 and 1')
    return x


def is_number(x):
    """Argparse parsing function: returns `x` as an int or float if it is a number.

    Args:
        x (str): the command line argument to be checked.

    Returns:
        int|float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if x.is_integer():
        return int(x)
    return x


def is_percentage_or_minus_1(x):
    """Argparse parsing function: returns `x` as an int if it is -1 or float if between 0 and 1.

    Args:
        x (str): the command line argument to be checked.

    Returns:
        int|float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if x == -1:
        return int(x)
    if 0 <= x <= 1:
        return x
    raise argparse.ArgumentTypeError('Argument has to be either -1 or between 0 and 1')


# parsers for each callable script
parsers = dict()

# possible choices for categorical command-line arguments
# MAKE_DATASETS
data_choices = ['spark', 'otn']
threshold_sup_choices = ['supervised', 'unsupervised']

# BUILD_FEATURES
alter_bundles_choices = ['.', 'spark_bundles', 'otn_bundles']
transform_choices = [
    f'{s}{d}' for s in [
        'regular_scaling', 'trace_scaling', 'head_scaling', 'head_online_scaling'
    ] for d in ['', '.pca']
]
scaling_choices = ['minmax', 'std']
# when using `head_online_scaling`, the size of the rolling window is fixed to `head_size`
online_window_choices = ['expanding', 'rolling']

# TRAIN_MODEL
forecasting_choices = ['naive.forecasting', 'mlp', 'rnn']
reconstruction_choices = ['ae', 'bigan']
modeling_split_choices = ['random.split', 'stratified.split', 'service.split']
# model hyperparameters choices
model_choices = forecasting_choices + reconstruction_choices
unit_choices = ['rnn', 'lstm', 'gru']
opt_choices = ['sgd', 'adam', 'nadam', 'rmsprop']
ae_type_choices = ['dense', 'conv', 'rec']
enc_type_choices = gen_type_choices = dis_type_choices = ae_type_choices
ae_dec_last_act_choices = ['linear', 'sigmoid']
ae_loss_choices = ['mse', 'bce']
bigan_gen_last_act_choices = ['linear', 'sigmoid', 'tanh']

# TRAIN_SCORER
forecasting_scoring_choices = ['re', 'nll']
reconstruction_scoring_choices = ['mse', 'mse.dis', 'mse.ft']
scoring_choices = forecasting_scoring_choices + reconstruction_scoring_choices
# use regular, range-based, or 'fast-alert' Precision and Recall
evaluation_choices = ['regular', 'range', 'fast.alert']
omega_choices = ['default', 'flat_scale']
delta_choices = ['flat', 'front', 'back']
# fully allow duplicates, fully penalize them or penalize them using an inverse polynomial penalty
gamma_choices = ['dup', 'no.dup', 'inv.poly']

# TRAIN_DETECTOR
# methods for selecting the outlier score threshold
supervised_ts_sel_choices = ['smallest', 'largest', 'brent', 'search']
two_stat_ts_sel_choices = ['std', 'mad', 'iqr']
unsupervised_ts_sel_choices = two_stat_ts_sel_choices
# supervised threshold selection target if multiple event types (avg or global performance)
sup_threshold_target_choices = ['avg.perf', 'global.perf']

# TRAIN_EXPLAINER
model_free_explanation_choices = ['exstream', 'macrobase']
model_based_explanation_choices = ['lime']
explanation_choices = model_free_explanation_choices + model_based_explanation_choices
explained_predictions_choices = ['ground.truth', 'model']

# RUN_PIPELINE
pipeline_type_choices = ['ad', 'ed', 'ad.ed']

# arguments for `make_datasets.py`
parsers['make_datasets'] = argparse.ArgumentParser(
    description='Make train[, threshold selection] and test sets from the input data', add_help=False
)
parsers['make_datasets'].add_argument(
    '--data', default=DEFAULTS['data'], choices=data_choices,
    help='data to use as input to the pipeline'
)
# outlier score threshold selection (relevant here for constituting the datasets)
parsers['make_datasets'].add_argument(
    '--threshold-supervision', default=DEFAULTS['threshold_supervision'],
    choices=threshold_sup_choices,
    help='threshold selection supervision (`supervised` or `unsupervised`)'
)
# trace labelling: 3 consecutive intervals for each labeled anomaly
parsers['make_datasets'].add_argument(
    '--primary-start', default=DEFAULTS['primary_start'],
    help='start of the primary target period (relative to an anomaly timestamp)'
)
parsers['make_datasets'].add_argument(
    '--secondary-start', default=DEFAULTS['secondary_start'],
    help='start of the secondary target period'
)
parsers['make_datasets'].add_argument(
    '--secondary-end', default=DEFAULTS['secondary_end'],
    help='end of the secondary target period'
)
parsers['make_datasets'].add_argument(
    '--tolerance-end', default=DEFAULTS['tolerance_end'],
    help='end of the tolerance zone (where positive predictions are not penalized)'
)
# pre-resampling for more efficient storage
parsers['make_datasets'].add_argument(
    '--pre-sampling-period', default=DEFAULTS['pre_sampling_period'],
    help='possible pre-redefinition of the sampling period to save space'
)

# additional arguments for `build_features.py`
parsers['build_features'] = argparse.ArgumentParser(
    parents=[parsers['make_datasets']], description='Build features for the model inputs', add_help=False
)
parsers['build_features'].add_argument(
    '--sampling-period', default=DEFAULTS['sampling_period'],
    help='the records will be downsampled to the provided period'
)
parsers['build_features'].add_argument(
    '--alter-bundles', default=DEFAULTS['alter_bundles'], choices=alter_bundles_choices,
    help='list of features alteration bundles we want to choose from (`.` for none)'
)
parsers['build_features'].add_argument(
    '--alter-bundle-idx', default=DEFAULTS['alter_bundle_idx'], type=int,
    help='alteration bundle index in the used bundles list'
)
parsers['build_features'].add_argument(
    '--transform-chain', default=DEFAULTS['transform_chain'], choices=transform_choices,
    help='features transformation chain, dot-separated (`.` for no transformation)'
)
parsers['build_features'].add_argument(
    '--head-size', default=DEFAULTS['head_size'], type=int,
    help='number of records used at the beginning of each trace to train a transformer'
)
parsers['build_features'].add_argument(
    '--online-window-type', default=DEFAULTS['online_window_type'], choices=online_window_choices,
    help='whether to use an expanding or rolling window when using head-online scaling'
)
parsers['build_features'].add_argument(
    '--regular-pretraining-weight', default=DEFAULTS['regular_pretraining_weight'],
    type=is_percentage_or_minus_1,
    help='if not -1, use regular pretraining for head/head-online transformers with this weight'
)
parsers['build_features'].add_argument(
    '--scaling-method', default=DEFAULTS['scaling_method'], choices=scaling_choices,
    help='(re)scaling method (standard or min-max)'
)
parsers['build_features'].add_argument(
    '--minmax-range', default=DEFAULTS['minmax_range'], nargs='+', type=int,
    help='range of output features if using minmax scaling'
)
parsers['build_features'].add_argument(
    '--n-components', default=DEFAULTS['n_components'], type=is_number,
    help='number of components or percentage of explained variance for PCA'
)

# additional arguments for `train_model.py`
parsers['train_model'] = argparse.ArgumentParser(
    parents=[parsers['build_features']], description='Train ML model to perform a downstream task', add_help=False
)
# train/val/test datasets constitution for the modeling task
parsers['train_model'].add_argument(
    '--modeling-split', default=DEFAULTS['modeling_split'], choices=modeling_split_choices,
    help='splitting strategy for constituting the modeling `train/val/test` datasets'
)
parsers['train_model'].add_argument(
    '--modeling-val-prop', default=DEFAULTS['modeling_val_prop'], type=float,
    help='proportion of `train` going to the modeling validation dataset'
)
parsers['train_model'].add_argument(
    '--modeling-test-prop', default=DEFAULTS['modeling_test_prop'], type=float,
    help='proportion of `train` going to the modeling test dataset'
)
parsers['train_model'].add_argument(
    '--n-period-strata', default=DEFAULTS['n_period_strata'], type=int,
    help='number of bins per period for the stratified modeling split'
)
parsers['train_model'].add_argument(
    '--model-type', default=DEFAULTS['model_type'], choices=model_choices,
    help='type of machine learning model'
)
# forecasting-specific arguments
parsers['train_model'].add_argument(
    '--n-back', default=DEFAULTS['n_back'], type=int,
    help='number of records we can look back to perform forecasts'
)
parsers['train_model'].add_argument(
    '--n-forward', default=DEFAULTS['n_forward'], type=int,
    help='number of records we aim to forecast forward'
)
# reconstruction-specific arguments
parsers['train_model'].add_argument(
    '--window-size', default=DEFAULTS['window_size'], type=int,
    help='size of windows to reconstruct in number of records'
)
parsers['train_model'].add_argument(
    '--window-step', default=DEFAULTS['window_step'], type=int,
    help='number of records between each extracted training window'
)
# whether to tune hyperparameters using Tree Parzen Estimator (TPE)
parsers['train_model'].add_argument(
    '--hyperas-tuning', action='store_true',
    help='if provided, tune hyperparameters using hyperas\' TPE'
)
# whether to tune hyperparameters using keras tuner (preferred method)
parsers['train_model'].add_argument(
    '--keras-tuning', action='store_true',
    help='if provided, tune hyperparameters using keras tuner'
)
# FORECASTING MODELS #
# RNN hyperparameters
parsers['train_model'].add_argument(
    '--rnn-n-hidden-neurons', default=DEFAULTS['rnn_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each hidden layer of the RNN (before regression)'
)
parsers['train_model'].add_argument(
    '--rnn-unit-type', default=DEFAULTS['rnn_unit_type'], choices=unit_choices,
    help='type of recurrent units used by the network'
)
parsers['train_model'].add_argument(
    '--rnn-dropout', default=DEFAULTS['rnn_dropout'], type=float,
    help='dropout rate for feed-forward layers'
)
parsers['train_model'].add_argument(
    '--rnn-rec-dropout', default=DEFAULTS['rnn_rec_dropout'], type=float,
    help='dropout rate for recurrent layers'
)
# RECONSTRUCTION MODELS #
# Autoencoder hyperparameters
parsers['train_model'].add_argument(
    '--ae-latent-dim', default=DEFAULTS['ae_latent_dim'], type=int,
    help='dimension of windows latent representation'
)
parsers['train_model'].add_argument(
    '--ae-type', default=DEFAULTS['ae_type'], choices=ae_type_choices,
    help='type of autoencoder to use (dense, convolutional or recurrent)'
)
parsers['train_model'].add_argument(
    '--ae-enc-n-hidden-neurons', default=DEFAULTS['ae_enc_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each hidden layer of the encoder (before the coding)'
)
parsers['train_model'].add_argument(
    '--ae-dec-last-activation', default=DEFAULTS['ae_dec_last_activation'],
    choices=ae_dec_last_act_choices, help='activation function of the last decoder layer'
)
parsers['train_model'].add_argument(
    '--ae-dropout', default=DEFAULTS['ae_dropout'], type=float,
    help='dropout rate for feed-forward layers'
)
parsers['train_model'].add_argument(
    '--ae-rec-unit-type', default=DEFAULTS['ae_rec_unit_type'], choices=unit_choices,
    help='type of recurrent units used by the autoencoder network (if recurrent)'
)
parsers['train_model'].add_argument(
    '--ae-rec-dropout', default=DEFAULTS['ae_rec_dropout'], type=float,
    help='dropout rate for recurrent layers'
)
parsers['train_model'].add_argument(
    '--ae-loss', default=DEFAULTS['ae_loss'], choices=ae_loss_choices,
    help='loss function of the autoencoder network'
)
# BiGAN hyperparameters
parsers['train_model'].add_argument(
    '--bigan-latent-dim', default=DEFAULTS['bigan_latent_dim'], type=int,
    help='dimension of windows latent representation'
)
parsers['train_model'].add_argument(
    '--bigan-dis-threshold', default=DEFAULTS['bigan_dis_threshold'], type=float,
    help='only update D if its loss was above this value on the previous batch'
)
for name, ch in zip(
        ['encoder', 'generator', 'discriminator'],
        [enc_type_choices, gen_type_choices, dis_type_choices]
):
    abv = name[:3]
    parsers['train_model'].add_argument(
        f'--bigan-{abv}-type', default=DEFAULTS[f'bigan_{abv}_type'], choices=ch,
        help=f'type of {name} to use (dense, convolutional or recurrent)'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{abv}-arch-idx', default=DEFAULTS[f'bigan_{abv}_arch_idx'], type=int,
        help=f'if not -1, index of the architecture to use for this {name} type'
    )
    a_abv, d_abv = abv, abv
    if abv == 'dis':
        a_abv, d_abv, name = 'dis-x', 'dis_x', 'x path of D'
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-n-hidden-neurons',
        default=DEFAULTS[f'bigan_{d_abv}_rec_n_hidden_neurons'], nargs='+', type=int,
        help=f'number of neurons for each recurrent layer of {name}'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-unit-type',
        default=DEFAULTS[f'bigan_{d_abv}_rec_unit_type'], choices=unit_choices,
        help=f'type of recurrent units used by the {name} (if recurrent)'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-conv-n-filters',
        default=DEFAULTS[f'bigan_{d_abv}_conv_n_filters'], type=int,
        help=f'initial number of filters used by the {name} (if convolutional)'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-dropout', default=DEFAULTS[f'bigan_{d_abv}_dropout'], type=float,
        help=f'dropout rate for feed-forward layers of the {name}'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-dropout',
        default=DEFAULTS[f'bigan_{d_abv}_rec_dropout'], type=float,
        help=f'dropout rate for recurrent layers of the {name}'
    )
parsers['train_model'].add_argument(
    '--bigan-gen-last-activation', default=DEFAULTS['bigan_gen_last_activation'],
    choices=bigan_gen_last_act_choices, help='activation function of the last generator layer'
)
# z path of the discriminator
parsers['train_model'].add_argument(
    '--bigan-dis-z-n-hidden-neurons',
    default=DEFAULTS['bigan_dis_z_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each layer of the z path of D'
)
parsers['train_model'].add_argument(
    '--bigan-dis-z-dropout', default=DEFAULTS['bigan_dis_z_dropout'], type=float,
    help='dropout rate for feed-forward layers of the z path of D'
)

# SHARED WITH DIFFERENT PREFIXES #
for mt in ['rnn', 'ae', 'bigan']:
    if mt == 'bigan':
        for n in ['dis', 'enc_gen']:
            dashed_n = n.replace('_', '-')
            parsers['train_model'].add_argument(
                f'--{mt}-{dashed_n}-optimizer', default=DEFAULTS[f'{mt}_{n}_optimizer'], choices=opt_choices,
                help='optimization algorithm used for training the network'
            )
            parsers['train_model'].add_argument(
                f'--{mt}-{dashed_n}-learning-rate', default=DEFAULTS[f'{mt}_{n}_learning_rate'], type=float,
                help='learning rate used by the optimization algorithm'
            )
    else:
        parsers['train_model'].add_argument(
            f'--{mt}-optimizer', default=DEFAULTS[f'{mt}_optimizer'], choices=opt_choices,
            help='optimization algorithm used for training the network'
        )
        parsers['train_model'].add_argument(
            f'--{mt}-learning-rate', default=DEFAULTS[f'{mt}_learning_rate'], type=float,
            help='learning rate used by the optimization algorithm'
        )
    parsers['train_model'].add_argument(
        f'--{mt}-n-epochs', default=DEFAULTS[f'{mt}_n_epochs'], type=int,
        help='maximum number of epochs to train the network for'
    )
    parsers['train_model'].add_argument(
        f'--{mt}-batch-size', default=DEFAULTS[f'{mt}_batch_size'], type=int,
        help='batch size used for training the network'
    )

# additional arguments for `train_scorer.py`
parsers['train_scorer'] = argparse.ArgumentParser(
    parents=[parsers['train_model']], description='Derive outlier scores from a trained model', add_help=False
)
parsers['train_scorer'].add_argument(
    '--scoring-method', default=DEFAULTS['scoring_method'], choices=scoring_choices,
    help='outlier score derivation method'
)
parsers['train_scorer'].add_argument(
    '--mse-weight', default=DEFAULTS['mse_weight'], type=float,
    help='MSE weight if the outlier score is a convex combination of the MSE and another loss'
)
# parameters defining the Precision, Recall and F-score
parsers['train_scorer'].add_argument(
    '--evaluation-type', default=DEFAULTS['evaluation_type'], choices=evaluation_choices,
    help='whether to evaluate ability to detect full ranges or send accurate alerts'
)
parsers['train_scorer'].add_argument(
    '--beta', default=DEFAULTS['beta'], type=float,
    help='F-Score parameter (relative importance of Recall vs. Precision)'
)
parsers['train_scorer'].add_argument(
    '--recall-alpha', default=DEFAULTS['recall_alpha'], type=float,
    help=f'existence reward factor for range-based Recall'
)
for metric in ['recall', 'precision']:
    metric_text = metric.capitalize()
    f_choices = [omega_choices, delta_choices, gamma_choices]
    for f_name, f_desc, choices in zip(['omega', 'delta', 'gamma'], ['size', 'bias', 'cardinality'], f_choices):
        parsers['train_scorer'].add_argument(
            f'--{metric}-{f_name}', default=DEFAULTS[f'{metric}_{f_name}'], choices=choices,
            help=f'{f_desc} function for range-based {metric_text}'
        )

# additional arguments for `train_detector.py`
parsers['train_detector'] = argparse.ArgumentParser(
    parents=[parsers['train_scorer']],
    description='Select the outlier score threshold and derive binary predictions', add_help=False
)
# threshold selection parameters
parsers['train_detector'].add_argument(
    '--threshold-selection', default=DEFAULTS['threshold_selection'], nargs='+',
    help='list of outlier score threshold selection methods to try'
)
# supervised threshold selection parameters
# TODO - for now this parameter is unused in the pipeline (fixed to `avg.perf`)
parsers['train_detector'].add_argument(
    '--supervised-threshold-target', default=DEFAULTS['supervised_threshold_target'],
    choices=sup_threshold_target_choices,
    help='threshold selection target if multiple event types (`avg.perf` of `global.perf`)'
)
parsers['train_detector'].add_argument(
    '--n-bins', default=DEFAULTS['n_bins'], nargs='+', type=int,
    help='number of equal-sized bins for the `search` threshold selection method (list to try)'
)
parsers['train_detector'].add_argument(
    '--n-bin-trials', default=DEFAULTS['n_bin_trials'], nargs='+', type=int,
    help='number of times to evaluate the F-score within each bin (list to try)'
)
# unsupervised threshold selection parameters
parsers['train_detector'].add_argument(
    '--thresholding-factor', default=DEFAULTS['thresholding_factor'], nargs='+', type=float,
    help='`ts = stat_1 + thresholding_factor * stat_2` for simple unsupervised methods (list to try)'
)
parsers['train_detector'].add_argument(
    '--n-iterations', default=DEFAULTS['n_iterations'], nargs='+', type=int,
    help='number of thresholding iterations, each time removing the most obvious outliers (list to try)'
)
parsers['train_detector'].add_argument(
    '--removal-factor', default=DEFAULTS['removal_factor'], nargs='+', type=float,
    help='scores above `removal_factor * ts@{iteration_i}` will be removed for iteration i+1 (list to try)'
)
# additional arguments for `train_explainer.py`
parsers['train_explainer'] = argparse.ArgumentParser(
    parents=[parsers['train_detector']],
    description='Train an explanation discovery model to explain positive predictions', add_help=False
)
parsers['train_explainer'].add_argument(
    '--explanation-method', default=DEFAULTS['explanation_method'], choices=explanation_choices,
    help='explanation discovery method'
)
parsers['train_explainer'].add_argument(
    '--explained-predictions', default=DEFAULTS['explained_predictions'], choices=explained_predictions_choices,
    help='positive predictions to explain (ground truth or outputs of a model)'
)
# evaluation parameters
parsers['train_explainer'].add_argument(
    '--exp-eval-min-sample-length', default=DEFAULTS['exp_eval_min_sample_length'], type=int,
    help='minimum length of an explanation sample for it to be counted in the evaluation'
)
parsers['train_explainer'].add_argument(
    '--exp-eval-min-anomaly-length', default=DEFAULTS['exp_eval_min_anomaly_length'], type=int,
    help='minimum length of an anomaly for it to be counted in the evaluation'
)
parsers['train_explainer'].add_argument(
    '--exp-eval-n-runs', default=DEFAULTS['exp_eval_n_runs'], type=int,
    help='number of evaluation runs for a sample\'s explanation'
)
parsers['train_explainer'].add_argument(
    '--exp-eval-test-prop', default=DEFAULTS['exp_eval_test_prop'], type=is_percentage,
    help='proportion of records used as test for evaluating the explanation of a sample'
)
# EXstream hyperparameters
parsers['train_explainer'].add_argument(
    '--exstream-tolerance', default=DEFAULTS['exstream_tolerance'], type=float,
    help='tolerance hyperparameter for the EXstream explanation discovery method'
)
parsers['train_explainer'].add_argument(
    '--exstream-correlation-threshold', default=DEFAULTS['exstream_correlation_threshold'], type=float,
    help='correlation threshold hyperparameter for the EXstream explanation discovery method'
)
# MacroBase hyperparameters
parsers['train_explainer'].add_argument(
    '--macrobase-min-risk-ratio', default=DEFAULTS['macrobase_min_risk_ratio'], type=float,
    help='minimum risk ratio for the MacroBase explanation discovery method'
)
parsers['train_explainer'].add_argument(
    '--macrobase-n-bins', default=DEFAULTS['macrobase_n_bins'], type=int,
    help='number of bins for the MacroBase explanation discovery method'
)
parsers['train_explainer'].add_argument(
    '--macrobase-min-support', default=DEFAULTS['macrobase_min_support'], type=is_percentage,
    help='minimum support for the MacroBase explanation discovery method'
)
# LIME hyperparameters
parsers['train_explainer'].add_argument(
    '--lime-n-features', default=DEFAULTS['lime_n_features'], type=int,
    help='number of features in the explanation of LIME'
)
# additional arguments for `run_pipeline.py`
parsers['run_pipeline'] = argparse.ArgumentParser(
    parents=[parsers['train_explainer']], description='Run a complete pipeline', add_help=False
)
parsers['run_pipeline'].add_argument(
    '--pipeline-type', default=DEFAULTS['pipeline_type'], choices=pipeline_type_choices,
    help='type of pipeline to run (AD only, ED only or AD + ED)'
)

# add data-specific arguments
add_specific_args = importlib.import_module(f'utils.{os.getenv("USED_DATA").lower()}').add_specific_args
for k in 'make_datasets', 'build_features', 'train_model', 'train_scorer', 'train_detector', 'train_explainer':
    parsers = add_specific_args(parsers, k, PIPELINE_STEPS)

# add back `help` arguments to parent parsers
for key in \
        'make_datasets', 'build_features', 'train_model', 'train_scorer', 'train_detector', \
        'train_explainer', 'run_pipeline':
    parsers[key].add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit'
    )


def get_command_line_string(args_namespace, script_name):
    """Returns the subset of `args_namespace` defined for `script_name` as a full command-line string."""
    return get_str_from_formatted_args_dict(get_script_args_as_formatted_dict(args_namespace, script_name))


def get_script_args_as_formatted_dict(args_ns, script_name):
    """Returns the subset of `args_namespace` defined for `script_name` as a formatted dictionary.

    Args:
        args_ns (argparse.Namespace): the larger set of arguments, as outputted by `parse_args`.
        script_name (str): we want to restrict the arguments to the ones defined for this script name.

    Returns:
        dict: the restricted arguments, as a formatted dictionary.
            `key`: argument name in command-line format. Example: --my-arg-name.
            `value`: argument value.
            Note: the arguments with action='store_true' are handled in the same way as if they were
            entered by a user: `True` = present, `False` = absent.
    """
    # only keep arguments defined for the script name
    args_dict = {k: v for k, v in vars(args_ns).items() if k in [
        e.dest for e in parsers[script_name]._actions if e.dest != 'help'
    ]}
    # remove `False` arguments and empty out values of `True` arguments
    args_dict = get_handled_store_true(args_dict)
    # turn the dictionary keys into command-line argument names
    return {('--' + k).replace('_', '-'): v for k, v in args_dict.items()}


def get_handled_store_true(args_dict):
    """Returns the input arguments dictionary with handled arguments of type action='store_true'.

    Such arguments are either `True` or `False`, meaning they are either specified or not in the command-line call.
    We want to reproduce this behavior for the arguments dictionary.

    Args:
        args_dict (dict): original arguments dictionary.

    Returns:
        dict: the same dictionary with removed `False` arguments and emptied out `True` arguments.
    """
    # remove arguments whose values are `False`, since that means they were not specified
    args_dict = {k: v for k, v in args_dict.items() if not (type(v) == bool and not v)}
    # empty values of arguments whose values are `True`, since they should simply be mentioned without a value
    for k, v in args_dict.items():
        if type(v) == bool and v:
            args_dict[k] = ''
    return args_dict


def get_str_from_formatted_args_dict(args_dict):
    """Turns a formatted arguments dictionary into a command-line string, as if the arguments were entered by a user.

    Args:
        args_dict (dict): the arguments as a formatted dictionary.

    Returns:
        str: the corresponding command-line string.
    """
    args_str = str(args_dict)
    for c in '{}\',:[]':
        args_str = args_str.replace(c, '')
    return args_str
