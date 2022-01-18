import pickle
from sklearn.dummy import DummyRegressor


class BaselineModel:
    def __init__(self, strategy="mean", quantile=0.5):
        if strategy == "quantile":
            self.model = DummyRegressor(strategy=strategy, quantile=quantile)

        else:
            self.model = DummyRegressor(strategy=strategy)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def load(self, path):
        self.model = pickle.load(path)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, path)
