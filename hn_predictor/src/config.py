class TreeConfig:
    SEED = 42

    def __init__(self, verbose=1):
        self.verbose = verbose

    def params(self, **kwargs):
        params_dict = {
            "seed": self.SEED,
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "rmse",
            "metric": ["rmse", "mape"],
            "verbose": self.verbose,
        }
        params_dict.update(**kwargs)

        return params_dict
