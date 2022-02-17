import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)


def sample_wise_rsme(predictions, target):
    df = pd.concat([predictions, target], axis=1)
    df.columns = ["predictions", "target"]
    residuals = df["predictions"] - df["target"]
    residuals = (residuals ** 2) ** 0.5

    return residuals.values


def sample_wise_mape(predictions, target):
    df = pd.concat([predictions, target], axis=1)
    df.columns = ["predictions", "target"]
    residuals = np.abs(df["predictions"] - df["target"]) / df["target"]

    return residuals.values


def compute_sample_wise_metrics(predictions, target):
    return {
        "rmse": sample_wise_rsme(predictions, target),
        "mape": sample_wise_mape(predictions, target),
    }


def compute_metrics(predictions, target):
    return {
        "rmse": mean_squared_error(target, predictions, squared=False),
        "mape": mean_absolute_percentage_error(target, predictions),
        "explained_variance_score": explained_variance_score(target, predictions),
    }
