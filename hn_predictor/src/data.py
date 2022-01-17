import os
from unittest import expectedFailure
import pandas as pd
from pandas_profiling import ProfileReport
import sweetviz as sv

OVERWRITE_STRATEGY = os.getenv("COMET_ARTIFACT_OVERWRITE_STRATEGY", "OVERWRITE")


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


def profile_data(experiment, dataframes, minimal=True, log_raw_dataframe=False):
    for df, label in dataframes:
        experiment.log_dataframe_profile(
            df, label, minimal=minimal, log_raw_dataframe=log_raw_dataframe
        )
