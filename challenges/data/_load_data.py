import pandas as pd
from pathlib import Path

from .custom_dataprep import *

def load_data(file_path:Path, debug:bool = False):
    """
    Reads the data in file_path and returns is as pandas dataframe.
    Some initial data trasnformations are applied, such as NA removal.

    NOTE: this only works for the provided 'diamonds.csv' dataset.

    Parameters:
        - file_path (pathlib.Path): file path to diamonds.csv
        - debug (bool) (optional):  enable debug mode (default is False)

    Returns:
        - dataframe:    pandas dataframe with diamonds.csv dataset
    """
    # assert that file has correct extension
    assert file_path.suffix in ['.csv', '.txt']

    # load file into pandas dataframe
    data = pd.read_csv(file_path, header = 0)
    in_len = len(data)

    # remove NAs and price errors (if present)
    data = data.dropna(axis = 0)
    data = data[(data['x'] * data['y'] * data['z'] != 0) & (data['price'] > 0)]
    if debug: print(f'Removed {in_len - len(data)} points from the dataset due to missing or incorrect values.')

    return data
