import comet_ml

import os
import argparse

from utils import (
    load_data,
    fetch_dataset_artifact,
    sample_dataset,
    preprocess,
    profile_data,
    upload_dataset_artifact,
)

DATA_PATH = os.getenv("DATA_PATH", "../data")
ASSET_PATH = os.getenv("ASSET_PATH", "../assets")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_artifact_name", type=str)
    parser.add_argument("--input_artifact_version", type=str, default="latest")
    parser.add_argument("--output_artifact_name", type=str)
    parser.add_argument("--output_artifact_version", type=str)
    parser.add_argument("--target_name", type=str)
    parser.add_argument(
        "--pipeline_model_name", type=str, default="distilbert-base-uncased"
    )
    parser.add_argument("--pipeline_batch_size", type=int, default=8)
    parser.add_argument("--text_feature_name", type=str)
    parser.add_argument("--sampling_fraction", type=float, default=0.05)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--message", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment = comet_ml.Experiment()
    experiment.add_tag("feature-engineering")

    dataset_config = fetch_dataset_artifact(
        experiment,
        args.input_artifact_name,
        args.input_artifact_version,
        output_path=DATA_PATH,
    )

    dataframes = []
    for split_name, path in dataset_config.items():
        df = load_data(path)
        dataframes.append((df, split_name))

    samples = []
    for df, split_name in dataframes:
        samples.append(
            (
                sample_dataset(
                    df,
                    args.target_name,
                    frac=args.sampling_fraction,
                    random_state=args.random_state,
                ),
                split_name,
            )
        )
    profile_data(experiment, samples)

    feature_filenames = []
    for df, split_name in samples:
        features = preprocess(
            df,
            args.target_name,
            model_name=args.pipeline_model_name,
            text_feature_name=args.text_feature_name,
            pipeline_batch_size=args.pipeline_batch_size,
        )

        filename = f"{ASSET_PATH}/{split_name}.pkl"
        features.to_pickle(filename)
        feature_filenames.append(filename)

    upload_dataset_artifact(
        experiment,
        feature_filenames,
        args.output_artifact_name,
        args.output_artifact_version,
    )


if __name__ == "__main__":
    main()
