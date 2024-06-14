import os
from pathlib import Path
import pickle
from argparse import ArgumentParser
import optuna

from config import *
from data import load_data
from models import RegressionModel

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def main():
    # parse arguments
    input_file, output_dir, model_dict, debug = _parse_args()
    base_model, dataprep_f, params = model_dict

    def objective(trial: optuna.trial.Trial) -> float:
        # Load the hyperparameters for xgboost
        model_params = {key: value(trial) if callable(value) else value for key, value in params.items()}

        param = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True
        }

        model = XGBRegressor(**param)

        # load data
        data = load_data(input_file, debug = False)
        X_train, X_val, y_train, y_val = dataprep_f(data, train_size = TRAIN_SPLIT, random_state = SEED)

        # train model
        # trained_model = model.train((X_train, y_train))
        model.fit(X_train, y_train)

        # test model and return mae
        # scores = trained_model.test((X_val, y_val), METRICS)
        # mae = scores['mae']
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters: ", study.best_params)

    model = RegressionModel(base_model(**study.best_params))
    data = load_data(input_file, debug = False)
    X_train, X_test, y_train, y_test = dataprep_f(data, train_size = TRAIN_SPLIT, random_state = SEED)
    trained_model = model.train((X_train, y_train))
    scores = trained_model.test((X_test, y_test), METRICS)
    print(f'Results for {model.name}')
    for k, v in scores.items():
        print(f'\t{k}:\t{v}')

    # create dict for saving model and relative performance metrics
    model_dict = {
        'name':   model.name,
        'model':        trained_model,
        'scores':      scores
    }
    # save model dict
    output_file = output_dir / f'{model.name}Model.pkl'
    with open(output_file, 'wb') as file:
        pickle.dump(model_dict, file)
    if debug: print(f'Saved model dict to {output_file}')
    
def _parse_args():
    # argument parser
    parser = ArgumentParser(description = 'main file for challenge 1')
    parser.add_argument('input_file', type = str, help = 'file path to diamonds.csv dataset')
    parser.add_argument('output_dir', type = str, help = 'output directory to save model')
    parser.add_argument('model', choices = list(ARGS_DICT.keys()), help = 'linear model to use')
    parser.add_argument('--debug', action = 'store_true', help = 'enable debug mode')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file): 
        raise FileNotFoundError(f'File {args.input_file} does not exist.')
    
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f'The nasty error! Directory {args.output_dir} does not exist. Please create one before passing it to the program.')

    return Path(args.input_file), Path(args.output_dir), ARGS_DICT[args.model], args.debug

if __name__ == '__main__':
    main()