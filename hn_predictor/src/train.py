import comet_ml

import os
import argparse

from data.utils import load_data, fetch_dataset_artifact
from models import TreeModel
from evaluate import evaluation_pipeline
from config import TreeConfig

DATA_PATH = os.getenv("DATA_PATH", "../data")
ASSET_PATH = os.getenv("ASSET_PATH", "../assets")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str, default="latest")
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def train_and_evaluate(
    experiment,
    X_train,
    y_train,
    model,
    evaluation_pipeline,
    X_valid=None,
    y_valid=None,
):

    model.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
    evaluation_pipeline(experiment, model, X=X_valid, y=y_valid)

    return model


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

    train_df = train_df.dropna(subset=[args.target_name])
    valid_df = valid_df.dropna(subset=[args.target_name])

    if args.smoke_test:
        train_df, valid_df = map(lambda x: x.sample(n=10), [train_df, valid_df])

    y_train = train_df.pop(args.target_name)
    y_valid = valid_df.pop(args.target_name)

    X_train = train_df
    X_valid = valid_df

    config = TreeConfig()

    model = TreeModel(params=config.params())
    model = train_and_evaluate(
        experiment,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        evaluation_pipeline=evaluation_pipeline,
    )

    # Save the Model
    model_path = os.path.join(ASSET_PATH, "model.pkl")
    model.save(model_path)
    experiment.log_model(
        "hn-gbdt-regression",
        model_path,
    )


if __name__ == "__main__":
    main()
