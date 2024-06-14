import pickle
from argparse import ArgumentParser

def load():
    # parse arguments
    model_to_load, debug = _parse_args()

    with open(model_to_load, 'rb') as file:
        model_dict = pickle.load(file)

    model_name = model_dict['name']
    model = model_dict['model']
    hyperparams = model_dict['hyperparams']
    scores = model_dict['scores']

    print(f'Loaded {model_name} model, with ', end = '')
    if hyperparams:
        print(f'hyper-parameters:')
        for k, v in hyperparams.items():
            print(f'\t{k}:\t{v}')
    else:
        print(f'default hyperparameters')
    print(f'Results::')
    for k, v in scores.items():
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