import comet_ml
import os
import pandas as pd
import numpy as np

from transformers import pipeline
from tqdm.auto import tqdm

OVERWRITE_STRATEGY = os.getenv("COMET_ARTIFACT_OVERWRITE_STRATEGY", "OVERWRITE")
RANDOM_SEED = os.getenv("RANDOM_SEED", 42)


def load_data(data_path):
    return pd.read_pickle(data_path)


def fetch_dataset_artifact(
    experiment, artifact_name, artifact_version, artifact_assets=[], output_path="./"
):
    artifact = experiment.get_artifact(
        artifact_name, workspace=experiment.workspace, version_or_alias=artifact_version
    )
    metadata = artifact.metadata

    if not artifact_assets:
        # Download all data from the Artifact
        artifact.download(output_path, overwrite_strategy=OVERWRITE_STRATEGY)

    else:
        # Download specific assets in the Artifact
        for asset in artifact_assets:
            artifact.get_asset(asset).download(
                output_path, overwrite_strategy=OVERWRITE_STRATEGY
            )

    output = {}
    for key, value in metadata["filenames"].items():
        output[key] = f"{output_path}/{value}"

    return output


def upload_dataset_artifact(experiment, filenames, artifact_name, artifact_version):
    artifact = comet_ml.Artifact(artifact_name, "dataset")
    for filename in filenames:
        artifact.add(filename)

    experiment.log_artifact(artifact)


def sample_dataset(df, target, frac=0.05, random_state=42):
    df["score_bins"] = pd.qcut(df[target], q=10, duplicates="drop")
    sample = df.groupby("score_bins").apply(
        lambda x: x.sample(frac=frac, random_state=random_state)
    )
    sample = sample.reset_index(level=0, drop=True)
    sample.pop("score_bins")

    return sample


def extract_timestamp_features(df):
    output = pd.DataFrame()

    output["hour"] = df["timestamp"].dt.hour
    output["day"] = df["timestamp"].dt.day
    output["day_of_the_week"] = df["timestamp"].dt.dayofweek
    output["month"] = df["timestamp"].dt.month
    output["year"] = df["timestamp"].dt.year

    output.index = df.index

    return output


def preprocess(
    df,
    target_name,
    model_name="distilbert-base-uncased",
    text_feature_name="title",
    pipeline_batch_size=8,
):
    target = df.pop(target_name)
    pipe = pipeline(
        "feature-extraction",
        model=model_name,
        tokenizer=model_name,
    )
    timestamp_features = extract_timestamp_features(df)

    def datagen(texts):
        idx = 0
        while idx < len(texts):
            yield texts[idx]
            idx += 1

    results = []
    text_dataset = df[text_feature_name].values.tolist()

    for result in tqdm(
        pipe(
            datagen(text_dataset),
            pipeline_batch_size=pipeline_batch_size,
            device=0,
        ),
        total=len(text_dataset),
    ):
        results.append(result)

    results = np.array(list(map(lambda x: np.mean(x, axis=1), results)))
    results = np.squeeze(results)

    columns = [f"feature_{i}" for i in range(results.shape[1])]
    text_features = pd.DataFrame(results, columns=columns)
    text_features.index = df.index
    output = pd.concat([timestamp_features, text_features], axis=1)
    output[target_name] = target

    return output


def profile_data(experiment, dataframes, minimal=True, log_raw_dataframe=False):
    for df, label in dataframes:
        experiment.log_dataframe_profile(
            df, label, minimal=minimal, log_raw_dataframe=log_raw_dataframe
        )
