import comet_ml

import os
import argparse

from data import load_data, fetch_artifact

DATA_PATH = os.getenv("DATA_PATH")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("prediction")

    path = fetch_artifact(
        DATA_PATH, args.train_artifact_name, args.train_artifact_version
    )
    df = load_data(path)

    # TODO: Run predictions on the given dataset

    return


if __name__ == "__main__":
    main()
