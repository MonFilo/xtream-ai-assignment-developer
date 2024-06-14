import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(file_path:Path):
    """
    Reads the data in file_path and returns is as pandas dataframe

    Parameters:
        - file_path (pathlib.Path): file path to diamonds.csv

    Returns:
        - dataframe (X):    dataframe of covariates
        - dataframe (y):    single-column dataframe of response
    """

    # assert that file has correct extension
    assert file_path.suffix in ['.csv', '.txt']

    # read file into pandas dataframe
    data = pd.read_csv(file_path, header = 0)

    # load covariates
    X = data.drop(columns = 'price', axis = 1)

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

if __name__ == '__main__':
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parents[1]
    csv_file_path = parent_dir / 'data' / 'diamonds.csv'  # Construct the path to the CSV file
    X, y = load_data(csv_file_path)
    X_train, X_test, y_train, y_test = split(X, y, train_size = 0.8, random_state = 48)
    print(y_test)