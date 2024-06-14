import numpy as np
import pandas as pd
from pathlib import Path
import pickle

model_to_load = Path('data/saved_models/xgboost354.95.pkl')
with open(model_to_load, 'rb') as file:
    model_dict = pickle.load(file)
model_name = model_dict['name']
model = model_dict['model']
hyperparams = model_dict['hyperparams']
scores = model_dict['scores']

features = np.array([[
        1.01,
        'Premium',
        'E',
        'SI1',
        59.6,
        59.0,
        6.53,
        4.46,
        3.87
    ]])
df = pd.DataFrame(features, columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
print(df.describe())

# prediction = model.predict(pd)

# print(df)
