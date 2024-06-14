import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Define your custom dataprep function if your model needs it.

It should return:
    - dataframe X_train:  dataframe NxD for training    (N number of observations, D number of covariates)
    - dataframe X_test:   dataframe NxD for testing     (N number of observations, D number of covariates)
    - dataframe y_train:  dataframe NxR for training    (N number of observations, D number of covariates)
    - dataframe y_test:   dataframe NxR for testing     (N number of observations, D number of covariates)
'''

# data preparation for linear models
def dataprep_linear(data, **kwargs):
    # drop irrelevant columns and collinear variables
    data = data.drop(columns = ['depth', 'table', 'y', 'z'])
    # create dummy variables
    data = pd.get_dummies(data, columns = ['cut', 'color', 'clarity'], drop_first=True)

    # split covariates and responses
    X = data.drop(columns = 'price')
    y = data['price']

    # split dataframe training and testing
    X_train, X_test, y_train, y_test  = train_test_split(X, y, **kwargs)

    # use log transformation to prevent negative predictions
    y_train = np.log(y_train)

    return X_train, X_test, y_train, y_test

dataprep_ridge = dataprep_linear

def dataprep_lasso(data, **kwargs):
    # drop irrelevant columns and collinear variables
    data = data.drop(columns = ['depth', 'table', 'y', 'z'])
    # create dummy variables
    data = pd.get_dummies(data, columns = ['cut', 'color', 'clarity'], drop_first=True)

    # split covariates and responses
    X = data.drop(columns = 'price')
    y = data['price']

    # split dataframe training and testing
    X_train, X_test, y_train, y_test  = train_test_split(X, y, **kwargs)

    return X_train, X_test, y_train, y_test

# data preparation for xgboost model
def dataprep_xgboost(data, **kwargs):
    # trees prefer ordinal variables to categorical ones
    data['cut'] = pd.Categorical(data['cut'],
                                 categories = ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                 ordered = True)
    data['color'] = pd.Categorical(data['color'],
                                   categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                   ordered = True)
    data['clarity'] = pd.Categorical(data['clarity'],
                                     categories = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                                     ordered = True)

    # split covariates and responses
    X = data.drop(columns = 'price')
    y = data['price']

    # split dataframe training and testing
    X_train, X_test, y_train, y_test  = train_test_split(X, y, **kwargs)

    # use log transformation to prevent negative predictions
    y_train = np.log(y_train)

    return X_train, X_test, y_train, y_test
