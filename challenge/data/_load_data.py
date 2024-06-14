import pandas as pd
import pathlib as Path
from sklearn.model_selection import train_test_split

def load_data(file_path:Path):
    """
    Reads the data in file_path and returns is as pandas dataframe
    """
    assert file_path.suffix in ['.csv', '.txt']
    data = pd.read_csv(file_path, header = 0)
    print(data.head())
    X = data.drop(columns = 'price', axis = 1)
    y = data['price']
    return X, y

def split(dataframe:pd.DataFrame):
    """
    """
    df_train, df_val, df_test = train_test_split()

if __name__ == '__main__':
    current_dir = Path.PurePath(__file__).parent
    parent_dir = current_dir.parents[1]
    csv_file_path = parent_dir / 'data' / 'diamonds.csv'  # Construct the path to the CSV file
    X, y = load_data(csv_file_path)