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
    parser.add_argument("--smoke_test_frac", type=float, default=0.01)
    parser.add_argument("--optimizer_config", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def run_optimizer(
    optimizer_config,
    X_train,
    y_train,
    X_valid,
    y_valid,
    evaluation_pipeline,
):
    optimizer = comet_ml.Optimizer(optimizer_config)
    for experiment in optimizer.get_experiments():
        parameters = {
            "n_estimators": experiment.get_parameter("n_estimators"),
            "num_leaves": experiment.get_parameter("num_leaves"),
            "learning_rate": experiment.get_parameter("learning_rate"),
        }

        config = TreeConfig()
        model = TreeModel(params=config.params(**parameters))

        train_and_evaluate(
            experiment,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            evaluation_pipeline=evaluation_pipeline,
        )


def train_and_evaluate(
    experiment,
    X_train,
    y_train,
    model,
    evaluation_pipeline,
    X_valid=None,
    y_valid=None,
):
    experiment.add_tag("train_and_evaluate")
    experiment.add_tag("regression")

    model.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
    evaluation_pipeline(experiment, model, X=X_valid, y=y_valid)

    # Save the Model
    model_path = os.path.join(ASSET_PATH, "model.pkl")
    model.save(model_path)
    experiment.log_model(
        "hn-gbdt-regression",
        model_path,
    )


def main():
    args = get_args()

    experiment = comet_ml.Experiment()

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
        experiment.add_tag("smoke-test")
        train_df, valid_df = map(
            lambda x: x.sample(frac=args.smoke_test_frac), [train_df, valid_df]
        )

    y_train = train_df.pop(args.target_name)
    y_valid = valid_df.pop(args.target_name)

    X_train = train_df
    X_valid = valid_df

    if args.optimizer_config:
        run_optimizer(
            args.optimizer_config,
            X_train,
            y_train,
            X_valid,
            y_valid,
            evaluation_pipeline,
        )

    else:
        config = TreeConfig()
        model = TreeModel(params=config.params())
        train_and_evaluate(
            experiment,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            evaluation_pipeline=evaluation_pipeline,
        )


if __name__ == "__main__":
    main()
