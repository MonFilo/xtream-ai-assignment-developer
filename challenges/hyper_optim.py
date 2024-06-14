import os
import numpy as np
from pathlib import Path
import pickle
from argparse import ArgumentParser
import optuna

from config import *
from data import load_data

from xgboost import XGBRegressor

def hyper_optim():
    # parse arguments
    input_file, output_dir, model_dict, n_trials, debug = _parse_args()
    model_name, regressor, params, dataprep_f, log_transform, hyper = model_dict

    if not hyper:
        raise RuntimeError('This model has no hyper-parameters to tune. Please provide them in file config.py')

    def objective(trial: optuna.trial.Trial):
        # Load model hyperparameters
        model_params = {key: value(trial) if callable(value) else value for key, value in hyper.items()}

        # load model with chosen hyperparameters
        model = regressor(**model_params)

        # load data
        data = load_data(input_file, debug = False)
        X_train, X_val, y_train, y_val = dataprep_f(data, train_size = TRAIN_SPLIT, random_state = SEED)

        # train model
        model.fit(X_train, y_train)

        # make prediction
        preds = model.predict(X_val)
        if log_transform: preds = np.exp(preds)

        # compute and return mae
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials = n_trials)
    best_params = study.best_params
    best_params.update(params)
    print(f'Best hyperparameters for {model_name}')
    for key, value in best_params.items():
        print(f'\t{key}:\t{value}')

    # define model with best hyper-parameters
    model = regressor(**best_params)
    # load data
    data = load_data(input_file, debug = False)
    X_train, X_test, y_train, y_test = dataprep_f(data, train_size = TRAIN_SPLIT, random_state = SEED)
    # fit model
    model.fit(X_train, y_train)
    # make predicionts
    preds = model.predict(X_test)
    # get scores
    scores = {}
    print(f'Results for {model_name}')
    for key, value in METRICS.items():
        s = value(y_test, preds)
        scores[key] = s
        print(f'\t{key}:\t{s}')

    # create dict for saving model and relative performance metrics
    model_dict = {
        'name':         model_name,
        'model':        model,
        'hyperparams':  best_params,
        'scores':       scores
    }

    # save model dict
    output_file = output_dir / f"{model_name}{scores['mae']:.2f}.pkl"
    with open(output_file, 'wb') as file:
        pickle.dump(model_dict, file)
    if debug: print(f'Saved model dict to {output_file}')
    
def _parse_args():
    # argument parser
    parser = ArgumentParser(description = 'main file for challenge 1')
    parser.add_argument('input_file', type = str, help = 'file path to diamonds.csv dataset')
    parser.add_argument('output_dir', type = str, help = 'output directory to save model')
    parser.add_argument('model', choices = list(ARGS_DICT.keys()), help = 'regression model to use')
    parser.add_argument('--n_trials', type = int, default = 100, help = 'number of trials to run optima for (default is 100)')
    parser.add_argument('--debug', action = 'store_true', help = 'enable debug mode')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file): 
        raise FileNotFoundError(f'File {args.input_file} does not exist.')
    
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f'The nasty error! Directory {args.output_dir} does not exist. Please create one before passing it to the program.')

    return Path(args.input_file), Path(args.output_dir), ARGS_DICT[args.model], args.n_trials, args.debug

if __name__ == '__main__':
    hyper_optim()