import comet_ml

import os
import argparse

from data import load_data, fetch_dataset_artifact
from models import BaselineModel
from metrics import compute_metrics

DATA_PATH = os.getenv("DATA_PATH", "../data")
ASSET_PATH = os.getenv("ASSET_PATH", "../assets")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str, default="latest")
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--strategy", type=str, default="mean")
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("train_and_evaluate")

    dataset_config = fetch_dataset_artifact(
        experiment,
        args.artifact_name,
        args.artifact_version,
        output_path=DATA_PATH,
    )

    train_df = load_data(dataset_config["train"])
    valid_df = load_data(dataset_config["valid"])

    model = BaselineModel(strategy=args.strategy, quantile=args.quantile)

    # Remove Target Column from DataFrame
    y_train = train_df.pop(args.target_name)
    # Fit the Model
    model.fit(train_df, y_train)

    with experiment.train():
        predictions = model.predict(train_df)
        metrics = compute_metrics(predictions, y_train)
        experiment.log_metrics(metrics)

    with experiment.validate():
        y_valid = valid_df.pop(args.target_name)

        predictions = model.predict(valid_df)
        metrics = compute_metrics(predictions, y_valid)
        experiment.log_metrics(metrics)

    # Save the Model
    model_path = os.path.join(ASSET_PATH, "model.pkl")
    model.save(model_path)
    experiment.log_model(
        "hn-baseline-regression",
        model_path,
    )


if __name__ == "__main__":
    main()
