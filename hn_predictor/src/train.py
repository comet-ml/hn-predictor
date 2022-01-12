import comet_ml

import os
import argparse

from data import load_data, fetch_artifact

DATA_PATH = os.getenv("DATA_PATH")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_artifact_name", type=str)
    parser.add_argument("--train_artifact_version", type=str)
    parser.add_argument("--valid_artifact_name", type=str)
    parser.add_argument("--valid_artifact_version", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("train_and_evaluate")

    train_path = fetch_artifact(
        DATA_PATH, args.train_artifact_name, args.train_artifact_version
    )
    train_df = load_data(train_path)

    valid_path = fetch_artifact(
        DATA_PATH, args.valid_artifact_name, args.valid_artifact_version
    )
    valid_df = load_data(valid_path)

    # TODO: Model training and evaluation

    return


if __name__ == "__main__":
    main()
