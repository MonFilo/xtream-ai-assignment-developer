
class MyLinearModel():
    def __init__(self, model):
        self.model = model

    def train(self, data):
        X, y = data
        self.model.fit(X,y)

        return self.model
