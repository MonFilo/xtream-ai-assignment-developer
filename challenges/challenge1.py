import os
from pathlib import Path
import pickle
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from config import *
from data import load_data, prepare_data
from models import RegressionModel

def main():
    # parse arguments
    input_file, output_dir, model_dict, debug = _parse_args()
    model, dataprep_f = model_dict

    # define model
    model = RegressionModel(model)

    # load data
    data = load_data(input_file, debug)
    X, y = dataprep_f(data)

    # split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = TRAIN_SPLIT, random_state = SEED)

    # train model
    trained_model = model.train((X_train, y_train))

    # test model

    metrics = trained_model.test((X_test, y_test), METRICS)

    # print results
    print(f'Results for {model.name}')
    for k, v in metrics.items():
        print(f'\t{k}:\t{v}')

    # create dict for saving model and relative performance metrics
    model_dict = {
        'name':   model.name,
        'model':        trained_model,
        'metrics':      metrics
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