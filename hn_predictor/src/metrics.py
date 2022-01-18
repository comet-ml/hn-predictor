from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)


def compute_metrics(predictions, target):
    return {
        "rmse": mean_squared_error(target, predictions) ** 0.5,
        "mape": mean_absolute_percentage_error(target, predictions),
        "explained_variance_score": explained_variance_score(target, predictions),
    }
