import os
import pandas as pd
import sweetviz as sv


def load_data(data_path):
    return pd.read_pickle(data_path)


def fetch_dataset_artifact(
    experiment, artifact_name, artifact_version, artifact_assets=[], output_path="./"
):
    artifact = experiment.get_artifact(artifact_name, artifact_version)
    metadata = artifact.metadata

    if not artifact_assets:
        # Download all data from the Artifact
        artifact.download(output_path)

    else:
        # Download specific assets in the Artifact
        for asset in artifact_assets:
            artifact.get_asset(asset).download(output_path)

    output = {}
    for key, value in metadata["filenames"].items():
        output[key] = f"{output_path}/{value}"

    return output


def profile_data(df, experiment):
    report = sv.analyze(df)
    report.log_comet(experiment)
