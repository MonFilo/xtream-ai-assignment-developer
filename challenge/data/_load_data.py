import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(file_path:Path, debug:bool = False):
    """
    Reads the data in file_path and returns is as pandas dataframe.

    NOTE: this only works for the provided 'diamonds' dataset.

    Parameters:
        - file_path (pathlib.Path): file path to diamonds.csv
        - debug (bool) (optional):  enable debug mode (default is False)

    Returns:
        - dataframe (X):    dataframe of covariates
        - dataframe (y):    single-column dataframe of response
    """

    # assert that file has correct extension
    assert file_path.suffix in ['.csv', '.txt']

    # read file into pandas dataframe
    data = pd.read_csv(file_path, header = 0)
    in_len = len(data)

    # remove NAs and price errors (if present)
    data = data.dropna(axis = 0)
    data = data[(data['x'] * data['y'] * data['z'] != 0) & (data['price'] > 0)]
    if debug: print(f'Removed {in_len - len(data)} points from the dataset due to missing or incorrect values.')

    # remove unecessary covariates
    data = data.drop(columns = ['depth', 'table', 'y', 'z'])

    # create dummy variables
    data = pd.get_dummies(data, columns = ['cut', 'color', 'clarity'], drop_first=True)

    # load covariates
    X = data.drop(columns = 'price')

    # load response
    y = data['price']

    return X, y


def split(X:pd.DataFrame, y:pd.DataFrame, **kwargs):
    """
    Split input dataframe into train, validation, and testing dataframes

    Parameters:
        - X (pandas.DataFrame): input dataframe of covariates to split
        - y (pandas.DataFrame): input dataframe of response to split
        - **kwargs: addittional named arguments for sl;earm train_test_split function

    Returns:
        - dataframe (X_train):    dataframe of covariates for training
        - dataframe (X_test):    dataframe of covariates for testing
        - dataframe (y_train):    dataframe of response for training
        - dataframe (y_test):    dataframe of response for testing
    """
    return train_test_split(X, y, **kwargs)