import comet_ml

import os
import argparse

from data import load_data, fetch_artifact
from hn_predictor.src.data import profile_data

DATA_PATH = os.getenv("DATA_PATH")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("profile")

    path = fetch_artifact(
        DATA_PATH, args.train_artifact_name, args.train_artifact_version
    )
    df = load_data(path)
    profile_data(df, experiment)

    return


if __name__ == "__main__":
    main()
