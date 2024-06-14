import pandas as pd
import numpy as np

# custom class for linear models
class RegressionModel():
    """
    Custom class for running linear models

    Attributes:
        model (sklearn.linear_model):   linear model of the class
        name (str): name of the linear model

    Methods:
        train:      train the model, returns trained model
        test:       test the model, returns performance metrics
        predict:    make prediction of input with model
    """
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__

    def train(self, data:tuple):
        """
        Method to train the model on given data

        Parameters:
            - data (tuple): tuple of (X, y), where X is the training data, and y the training labels

        Returns:
            - MyLinearModel (self): trained model
        """
        X, y = data
        self.model.fit(X,y)

        return self
    
    def test(self, data:tuple, evaluations:dict):
        """
        Method to test the model on given data

        Parameters:
            - data (tuple): tuple of (X, y), where X is the testing data, and y the testing labels
            - evalutaions (dict):   dictionary of performance metrics to use, which should be
                'metrics_name': sklearn.metrics function

                e.g.
                'r2':  sklearn.metrics.r2_score

        Returns:
            - dict (scores): dictionary with all performance metrics:
                'metrics_name': metrics score

                e.g.
                'r2':  0.9089680951938046
        """
        X, y = data
        preds = self.model.predict(X)

        # bad work-around
        if self.name in ['LinearRegression', 'Ridge', 'XGBRegressor']:
            preds = np.exp(preds)

        scores = {}
        for key, value in evaluations.items():
            scores[key] = f'{value(y, preds)}'
    
        return scores
    
    def predict(self, data:pd.DataFrame):
        """
        Method to make predictions with model on given data

        Parameters:
            - data (pandas DataFrame): data to predict

        Returns:
            - array:    predicitons of the model
        """
        X = data
        return self.model.predict(X)
    