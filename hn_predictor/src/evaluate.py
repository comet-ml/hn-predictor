import pandas as pd
from metrics import compute_sample_wise_metrics


def log_model_predictions(experiment, model, X, y, metric="rmse", top_n=100):
    predictions = model.predict(X)
    predictions = pd.DataFrame(predictions, index=X.index, columns=["predictions"])

    sample_wise_metrics = compute_sample_wise_metrics(predictions, y)
    sample_wise_df = pd.DataFrame.from_dict(sample_wise_metrics)
    sample_wise_df.index = X.index

    top_n_sample_metrics = sample_wise_df.nlargest(n=top_n, columns=[metric])
    top_n_predictions = predictions[predictions.index.isin(top_n_sample_metrics.index)]
    top_n_X = X[X.index.isin(top_n_sample_metrics.index)]
    top_n_y = y[y.index.isin(top_n_sample_metrics.index)]

    predictions_df = pd.concat(
        [top_n_X, top_n_y, top_n_predictions, top_n_sample_metrics], axis=1
    )
    experiment.log_table(
        "predictions.json", predictions_df, headers=False, **{"orient": "records"}
    )


def evaluation_pipeline(experiment, model, X, y):
    for func in [log_model_predictions]:
        func(experiment, model, X, y)
