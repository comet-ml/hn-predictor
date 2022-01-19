import comet_ml

import os
import argparse

from data import load_data, fetch_dataset_artifact
from models import BaselineClassifier
from metrics import compute_metrics

DATA_PATH = os.getenv("DATA_PATH", "../data")
ASSET_PATH = os.getenv("ASSET_PATH", "../assets")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str, default="latest")
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--strategy", type=str, default="prior")
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=10)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("train_and_evaluate")
    experiment.add_tag("classification")
    experiment.log_parameters(
        {"threshold": args.threshold, "random_state": args.random_state}
    )

    dataset_config = fetch_dataset_artifact(
        experiment,
        args.artifact_name,
        args.artifact_version,
        output_path=DATA_PATH,
    )

    train_df = load_data(dataset_config["train"])
    valid_df = load_data(dataset_config["valid"])

    train_df = train_df.dropna(subset=[args.target_name])
    valid_df = valid_df.dropna(subset=[args.target_name])

    model = BaselineClassifier(strategy=args.strategy, random_state=args.random_state)

    # Remove Target Column from DataFrame
    y_train = train_df.pop(args.target_name)
    y_train = model.target_transform(y_train)
    # Fit the Model
    model.fit(train_df, y_train)

    with experiment.train():
        predictions = model.predict(train_df)
        metrics = compute_metrics(predictions, y_train)
        experiment.log_metrics(metrics)

    with experiment.validate():
        y_valid = valid_df.pop(args.target_name)
        y_valid = model.target_transform(y_valid)

        predictions = model.predict(valid_df)
        metrics = compute_metrics(predictions, y_valid)
        experiment.log_metrics(metrics)

    # Save the Model
    model_path = os.path.join(ASSET_PATH, "model.pkl")
    model.save(model_path)
    experiment.log_model(
        "hn-baseline-classification",
        model_path,
    )


if __name__ == "__main__":
    main()
