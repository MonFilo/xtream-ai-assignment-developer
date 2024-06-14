import os
from pathlib import Path
import pickle
from argparse import ArgumentParser
import numpy as np

from config import *
from data import load_data

def main():
    # parse arguments
    input_file, output_dir, model_dict, debug = _parse_args()
    model_name, model, model_params, dataprep_f, log_transform, _ = model_dict
    model = model(**model_params)

    # load data
    data = load_data(input_file, debug)
    # prepare data for specific model and split it into training and testing
    X_train, X_test, y_train, y_test = dataprep_f(data, train_size = TRAIN_SPLIT, random_state = SEED)

    # train model
    model.fit(X_train, y_train)

    # test model
    preds = model.predict(X_test)
    if log_transform: preds = np.exp(preds)
    
    # get scores
    scores = {}
    print(f'Results for {model_name}')
    for key, value in METRICS.items():
        s = value(y_test, preds)
        scores[key] = s
        print(f'\t{key}:\t{s}')

    # create dict for saving model and relative performance scores
    model_dict = {
        'name':        model_name,
        'model':       model,
        'hyperparams': model_params,
        'scores':      scores
    }

    # save model dict
    output_file = output_dir / f"{model_name}{scores['mae']:.2f}$.pkl"
    with open(output_file, 'wb') as file:
        pickle.dump(model_dict, file)
    if debug: print(f'Saved model dict to {output_file}')
    
def _parse_args():
    # argument parser
    parser = ArgumentParser(description = 'main file for challenge 1')
    parser.add_argument('input_file', type = str, help = 'file path to diamonds.csv dataset')
    parser.add_argument('output_dir', type = str, help = 'output directory to save model')
    parser.add_argument('model', choices = list(ARGS_DICT.keys()), help = 'regression model to use')
    parser.add_argument('--debug', action = 'store_true', help = 'enable debug mode')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file): 
        raise FileNotFoundError(f'File {args.input_file} does not exist.')
    
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f'The nasty error! Directory {args.output_dir} does not exist. Please create one before passing it to the program.')

    return Path(args.input_file), Path(args.output_dir), ARGS_DICT[args.model], args.debug

if __name__ == '__main__':
    main()