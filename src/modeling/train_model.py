"""Model training module.

Trains a Machine Learning model for performing a task useful to anomaly detection.
"""
import os
import time

try:
    from hyperopt import tpe, Trials
    from hyperas import optim
except ModuleNotFoundError:
    print('No hyperas in the current environment.')

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import hyper_to_path, parsers, get_output_path, get_model_args
from utils.common import forecasting_choices, reconstruction_choices, get_modeling_task_string
from data.helpers import load_mixed_formats

from modeling.data_splitters import get_splitter_classes

from modeling.forecasting.forecasters import forecasting_classes
from modeling.forecasting.hyperas_rnn_tuning import data, create_model
from modeling.forecasting.evaluation import save_forecasting_evaluation

from modeling.reconstruction.reconstructors import reconstruction_classes
from modeling.reconstruction.evaluation import save_reconstruction_evaluation


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['train_model'].parse_args()

    # set input and output paths
    INFO_PATH = get_output_path(args, 'make_datasets')
    INPUT_PATH = get_output_path(args, 'build_features', 'data')
    OUTPUT_PATH = get_output_path(args, 'train_model', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_model', 'comparison')

    if args.hyperas_tuning:
        # optimize RNN hyperparameters using hyperas
        optim.minimize(
            model=create_model, data=data, algo=tpe.suggest, max_evals=15, trials=Trials(),
            data_args=(vars(args), INPUT_PATH, INFO_PATH, COMPARISON_PATH)
        )
    else:
        # load training periods and information
        files = load_mixed_formats([INPUT_PATH, INFO_PATH], ['train', 'train_info'], ['numpy', 'pickle'])

        # set the data splitter for constituting the train/val/test sets of the modeling task
        data_splitter = get_splitter_classes()[args.modeling_split](args)

        # forecasting models
        if args.model_type in forecasting_choices:
            # constitute the train/val/test sets for the forecasting task
            data = data_splitter.get_modeling_split(
                files['train'], files['train_info'], n_back=args.n_back, n_forward=args.n_forward
            )
            # define forecaster depending on the model type
            forecaster = forecasting_classes[args.model_type](args, OUTPUT_PATH)
            # either tune hyperparameters or train using full command arguments
            if args.keras_tuning:
                forecaster.tune_hp(data)
            else:
                # fit model to training data, validating on validation data
                forecaster.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
                # save the trained model training, validation and test evaluations for comparison
                config_name = hyper_to_path(
                    args.model_type, *get_model_args(args), time.strftime('run.%Y.%m.%d.%H.%M.%S')
                )
                save_forecasting_evaluation(
                    data, forecaster, get_modeling_task_string(args), config_name, COMPARISON_PATH
                )

        # reconstruction models
        if args.model_type in reconstruction_choices:
            # constitute the train/val/test sets for the reconstruction task
            data = data_splitter.get_modeling_split(
                files['train'], files['train_info'], window_size=args.window_size, window_step=args.window_step,
            )
            # define reconstructor depending on the model type
            reconstructor = reconstruction_classes[args.model_type](args, OUTPUT_PATH)
            # either tune hyperparameters or train using full command arguments
            if args.keras_tuning:
                reconstructor.tune_hp(data)
            else:
                # fit model to training data, validating on validation data
                reconstructor.fit(data['X_train'], data['X_val'])
                # save the trained model training, validation and test evaluations for comparison
                config_name = hyper_to_path(
                    args.model_type, *get_model_args(args), time.strftime('run.%Y.%m.%d.%H.%M.%S')
                )
                save_reconstruction_evaluation(
                    data, reconstructor, get_modeling_task_string(args), config_name, COMPARISON_PATH
                )
