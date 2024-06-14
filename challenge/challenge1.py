import pickle
from argparse import ArgumentParser
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error

from config import *
from data import load_data, split
from models import MyLinearModel

def main():
    # parse arguments
    input_file, output_dir, base_model, debug = _parse_args()

    X, y = load_data(input_file, debug)

    # split dataset into training and testing
    X_train, X_test, y_train, y_test = split(X, y, train_size = TRAIN_SPLIT, random_state = SEED)

    # load model
    model = MyLinearModel(base_model)

    # train model
    trained_model = model.train((X_train, y_train))

    # test model
    evals = {
            'r2': r2_score,
            'mae':  mean_absolute_error
        }
    scores = trained_model.test((X_test, y_test), evals)

    # print results
    if debug:
        print(f'Results for {model.name}')
        for k, v in scores.items():
            print(f'{k}:\t{v}')

    # create dict for saving model and relative performance metrics
    model_dict = {
        'model':    trained_model,
        'metrics':  scores
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
    parser.add_argument('model', choices = ['linear', 'ridge', 'lasso'], help = 'linear model to use')
    parser.add_argument('--debug', action = 'store_true', help = 'enable debug mode')
    args = parser.parse_args()

    models_dict = {
        'linear': LinearRegression(),
        'ridge':  Ridge(),
        'lasso':  Lasso(),
    }

    return args.input_file, args.output_dir, models_dict[args.model], args.debug

if __name__ == '__main__':
    main()