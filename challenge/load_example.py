import os
import pickle
from argparse import ArgumentParser

def load():
    # parse arguments
    model_to_load, debug = _parse_args()

    with open(model_to_load, 'rb') as file:
        model_dict = pickle.load(file)

    print(model_dict)
    model_name, model, metrics = model_dict['name'], model_dict['model'], model_dict['metrics']

    if debug:
        print(f'Loaded model {model_name}.')
        print(f'Results for {model_name}:')
        for k, v in metrics.items():
            print(f'\t{k}:\t{v}')

    # use loaded model
    model = ...

    
def _parse_args():
    # argument parser
    parser = ArgumentParser(description = 'example file to load a saved model')
    parser.add_argument('model_to_load', type = str, help = 'file .pth of model to load')
    parser.add_argument('--debug', action = 'store_true', help = 'enable debug mode')
    args = parser.parse_args()

    return args.model_to_load, args.debug

if __name__ == '__main__':
    load()