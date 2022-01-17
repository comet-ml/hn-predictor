import comet_ml

import os
import argparse

from data import load_data, fetch_dataset_artifact, profile_data

DATA_PATH = os.getenv("DATA_PATH", "../data")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--artifact_version", type=str, default="latest")
    parser.add_argument("--asset_names")
    parser.add_argument("--asset_split_name", type=str)
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("profile")

    data_config = fetch_dataset_artifact(
        experiment,
        artifact_name=args.artifact_name,
        artifact_version=args.artifact_version,
        output_path=DATA_PATH,
    )
    datasets = [
        [load_data(path), split_name] for split_name, path in data_config.items()
    ]
    # df = load_data(data_config[args.asset_split_name])
    profile_data(experiment, datasets, args.target_name)


if __name__ == "__main__":
    main()
