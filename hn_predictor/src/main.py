import os
from typing import Optional

# import comet_ml
import typer

from data import fetch_artifact, load_data, profile_data

DATA_PATH = os.getenv("DATA_PATH")

def main(
    message: Optional[str] = None,
    train_and_evaluate: bool = True,
    train_artifact_name: str ='train_predictions.pkl',
    validation_artifact_name: Optional[str] = 'validate_predictions.pkl',
    profile: bool = False,
    profile_artifact: Optional[str] = None,
    predict: bool = False,
    prediction_artifact: Optional[str] = None,
):
    '''Main entrypoint for training and predictions.

    Arguments:

    Options:

    '''

    if train_and_evaluate:
        train_artifact = fetch_artifact(train_artifact_name)
        validation_artifact = fetch_artifact(validation_artifact_name)\

    
    if profile:
        pass

    if predict:
        pass

    return


if __name__ == "__main__":
    typer.run(main)