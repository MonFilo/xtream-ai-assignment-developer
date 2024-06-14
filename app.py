from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from challenges.data import load_data

app = Flask(__name__)

# Load trained model
model_to_load = Path('data/saved_models/xgboost354.95.pkl')
with open(model_to_load, 'rb') as file:
    model_dict = pickle.load(file)
model_name = model_dict['name']
model = model_dict['model']
hyperparams = model_dict['hyperparams']
scores = model_dict['scores']

# Load  training dataset
input_file = Path('data/diamonds.csv')
training_data = load_data(input_file)    # TODO: only load training dataset

# predict price of diamond
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['carat'],
        data['cut'],
        data['color'],
        data['clarity'],
        data['depth'],
        data['table'],
        data['x'],
        data['y'],
        data['z']
    ]], dtype = object)

    df = pd.DataFrame(features, columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])

    df['cut'] = pd.Categorical(df['cut'],
                                 categories = ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                 ordered = True)
    df['color'] = pd.Categorical(df['color'],
                                   categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                   ordered = True)
    df['clarity'] = pd.Categorical(df['clarity'],
                                     categories = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                                     ordered = True)
    
    # predict price using trained model
    print(features)
    prediction = model.predict(df)
    print(prediction)

    return jsonify({'predicted_value': prediction})

# return n most similar diamonds of same cut, color, and clarity based on weight
@app.route('/similar', methods=['POST'])
def get_similar_samples():
    data = request.get_json()
    n_samples = data['n']
    cut = data['cut']
    color = data['color']
    clarity = data['clarity']
    weight = data['carat']
    
    # filter training data based on cut, color, and clarity
    filtered_data = training_data[(training_data['cut'] == cut) &
                                  (training_data['color'] == color) &
                                  (training_data['clarity'] == clarity)]
    
    # Use nearest neighbors to find the most similar samples by weight
    nn = NearestNeighbors(n_neighbors = n_samples)
    nn.fit(filtered_data[['carat']])
    # compute distances and indices of n nearest neighbors based on weight
    distances, indices = nn.kneighbors([[weight]])
    
    # 
    similar_samples = filtered_data.iloc[indices[0]]
    
    return similar_samples.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
