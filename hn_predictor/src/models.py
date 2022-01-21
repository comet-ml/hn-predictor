import pickle
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
import numpy as np


def threshold_factory(threshold=10):
    def _threshold(y):
        print(type(y))
        return np.array(y > threshold, dtype=np.int)
    return _threshold


class BaselineClassifier:
    def __init__(self, target_transform=None, strategy="prior", random_state=None):
        self.strategy = strategy
        self.random_state = random_state
        self.model = DummyClassifier(strategy=strategy, random_state=random_state)

        if target_transform is None:
            self.target_transform = lambda x: x
        else:
            self.target_transform = target_transform

    def fit(self, X, y):
        self.model.fit(X, self.target_transform(y))

    def predict(self, X):
        return self.model.predict(X)

    def load(self, path):
        self.model = pickle.load(path)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
