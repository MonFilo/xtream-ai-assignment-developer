from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error

from data import load_data, split
from models import MyLinearModel

# load data
parent_dir = Path(__file__).parents[1]     # get current directory
csv_file_path = parent_dir / 'data' / 'diamonds.csv'    # get data filepath

X, y = load_data(csv_file_path)

# split dataset into training and testing
X_train, X_test, y_train, y_test = split(X, y, train_size = 0.8, random_state = 48)

# load model
model = MyLinearModel(LinearRegression())

# train model
trained_model = model.train((X_train, y_train))

# test model
evals = {
        'r2': r2_score,
        'mae':  mean_absolute_error
    }
scores = trained_model.test((X_test, y_test), evals)

# print results
print(f'Results for {model.name}')
for k, v in scores.items():
    print(f'{k}:\t{v}')