import comet_ml

import os
import argparse

from data import load_data, fetch_dataset_artifact

DATA_PATH = os.getenv("DATA_PATH", "../data")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str)
    parser.add_argument("--asset_split_name", type=str)
    parser.add_argument("--asset_name", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("prediction")

    dataset_config = fetch_dataset_artifact(
        experiment,
        args.train_artifact_name,
        args.train_artifact_version,
        artifact_assets=[args.asset_name],
        output_path=DATA_PATH,
    )
    df = load_data(dataset_config[args.asset_split_name])

    # TODO: Run predictions on the given dataset

    return


if __name__ == "__main__":
    main()
