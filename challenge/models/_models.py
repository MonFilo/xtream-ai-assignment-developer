class MyLinearModel():
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__

    def train(self, data:tuple):
        X, y = data
        self.model.fit(X,y)

        return self
    
    def test(self, data:tuple, evaluations:dict):
        X, y = data
        preds = self.model.predict(X)
        scores = {}
        for key, value in evaluations.items():
            scores[key] = f'{value(y, preds)}'
    
        return scores
    
    def predict(self, data):
        X = data
        return self.model.predict(X)

if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_absolute_error

    model = LinearRegression()
    mymodel = MyLinearModel(model)
    trained_model = mymodel.train(([[1,1],[1,2]], [1,2]))
    print(trained_model.predict([[1,7]]))
    evals = {
        'r2': r2_score,
        'mae':  mean_absolute_error
    }

    scores = trained_model.test(([[1,4]], [4]), evals)
    for key, value in scores.items():
        print(f'{key}:\t{value}')