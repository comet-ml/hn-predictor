import lightgbm as lgb


class TreeModel:
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if not (X_valid.empty and y_valid.empty):
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        bst = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
        )
        self.model = bst

    def predict(self, X):
        return self.model.predict(X)

    def load(self, path):
        self.model = lgb.Booster(path)

    def save(self, path):
        self.model.save_model(path)
