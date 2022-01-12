import pandas as pd
import sweetviz as sv


def load_data(data_path):
    return pd.read_pickle(data_path)


def fetch_artifact(experiment, artifact_name, artifact_version, output_path):
    artifact = experiment.get_artifact(artifact_name, artifact_version)
    artifact.download(output_path)


def profile_data(df, experiment):
    report = sv.analyze(df)
    report.log_comet(experiment)
