def data(args_dict, input_path, info_path, comparison_path):
    import os
    import argparse
    import time

    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    from hyperopt import STATUS_OK
    from hyperas.distributions import choice, quniform, loguniform

    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from utils.common import hyper_to_path, get_output_path, get_modeling_task_string
    from data.helpers import load_mixed_formats
    from modeling.data_splitters import get_splitter_classes
    from modeling.forecasting.forecasters import build_rnn, RNN
    from modeling.forecasting.evaluation import save_forecasting_evaluation

    print('loading training periods and information...', end=' ', flush=True)
    files = load_mixed_formats([input_path, info_path], ['train', 'train_info'], ['numpy', 'pickle'])
    print('done.')
    print('creating training, validation and test (sequence, target) pairs...', end=' ', flush=True)
    splitter_args = {'model_type': args_dict['model_type']}
    if args_dict['modeling_split'] == 'stratified.split':
        splitter_args['n_period_strata'] = args_dict['n_period_strata']
    data_splitter = get_splitter_classes()[args_dict['modeling_split']](
        argparse.Namespace(**splitter_args), 0.15, 0.15
    )
    global data_dict
    data_dict = data_splitter.get_modeling_split(
        files['train'], files['train_info'], n_back=args_dict['n_back'], n_forward=args_dict['n_forward']
    )
    print('done.')
    global modeling_task_str
    modeling_task_str = get_modeling_task_string(argparse.Namespace(**args_dict))
    global args_d, comparison_pth
    args_d, comparison_pth = args_dict, comparison_path
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    return X_train, y_train, X_test, y_test


def create_model(X_train, y_train, X_test, y_test):
    n_hidden = {{choice([1, 2, 3])}}
    n_neurons = {{quniform(50, 200, 1)}}
    unit_type = {{choice(['lstm', 'gru'])}}
    dropout = {{choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])}}
    optimizer = {{choice(['nadam', 'adam', 'rmsprop'])}}
    learning_rate = {{loguniform(-10, -1)}}
    batch_size = {{choice([8, 16, 32, 64, 128])}}
    hp = {
        'n_hidden': n_hidden,
        'n_neurons': int(n_neurons),
        'unit_type': unit_type,
        'dropout': dropout,
        'recurrent_dropout': dropout,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }
    build_hp = {k: v for k, v in hp.items() if k != 'batch_size'}
    model = build_rnn((X_train.shape[1], X_train.shape[2]), y_train.shape[1], **build_hp)
    args = argparse.Namespace(**args_d)
    for k, v in hp.items():
        setattr(args, k, v)
    output_path = get_output_path(args, 'train_model')
    logging_path = os.path.join(output_path, time.strftime('%Y_%m_%d-%H_%M_%S'))
    tensorboard = TensorBoard(logging_path)
    checkpoint_a = ModelCheckpoint(os.path.join(output_path, 'model.h5'), save_best_only=True)
    checkpoint_b = ModelCheckpoint(os.path.join(logging_path, 'model.h5'), save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model.fit(
        X_train, y_train, batch_size=hp['batch_size'], epochs=100, verbose=1,
        validation_data=(data_dict['X_val'], data_dict['y_val']),
        callbacks=[tensorboard, checkpoint_a, checkpoint_b, early_stopping],
    )
    config_name = hyper_to_path(
        args.model_type, *list(hp.values()), time.strftime('run.%Y.%m.%d.%H.%M.%S')
    )
    evaluation_df = save_forecasting_evaluation(
        data_dict, RNN(args, output_path, model), modeling_task_str, config_name, comparison_pth
    )
    return {'loss': float(evaluation_df['VAL_MSE']), 'status': STATUS_OK, 'model': model}
