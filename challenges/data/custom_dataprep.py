import pandas as pd
'''
Define your custom dataprep function if your model needs it.

It should return:
    - dataframe X:  dataframe NxD    (N number of observations, D number of covariates)
    - dataframe y:  dataframe NxR    (N nubmer of observations, R number of response variables [1])
'''

# data preparation for linear models
def dataprep_linear(data):
    # drop irrelevant columns
    data = data.drop(columns = ['depth', 'table', 'y', 'z'])
    # create dummy variables since linear models cannot handle categorical variables
    data = pd.get_dummies(data, columns = ['cut', 'color', 'clarity'], drop_first=True)

    # split covariates and responses
    X = data.drop(columns = 'price')
    y = data['price']

    return X, y

# data preparation for xgboost model
def dataprep_xgboost(data):

    # split covariates and responses
    X = data.drop(columns = 'price')
    y = data['price']

    return X, y
