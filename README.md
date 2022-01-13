# hn-predictor

## Objective
The business objective that we are trying to achieve is to maximize the traffic we receive from our company’s post on Hacker News.

Our requirement is to create a model to predict the performance of a post on Hacker News.  Our model would help us optimize our post to maximize the likelihood that it will trend.

## Technical Requirements

Here we define the technical assumptions we are making for this project. These assumptions affect how code for running experiments should be written.

### Artifacts Requirements

#### Dataset Type Artifacts

For this project, all dataset type Artifacts should contain a training and validation dataset saved as pickle files. Artifact Metadata should follow this format.

```
{
    "filenames": {
        "train": "<name of train data pickle file>",
        "valid": "<name of validation data pickle file>",
    },
    "columns": {"<column name>": "<description of the data in the column>", ...},
}
```
The columns field should be a dictionary containing the name of the feature column as the key, and a description of the column as its value.

Our Dataset Artifact fetching functions will assume this schema and will not work otherwise.

### Experimentation Requirements

1. When running an experiment, pass in a message (similar to a git commit message) using the command line args in order to keep the project organized
2. When logging a prediction experiment, save the predictions as a `predictions.csv` and log them as an experiment asset
3. Log trained model assets using `experiment.log_model()` so that they can be added to the [Model Registry](https://www.comet.ml/site/using-comet-model-registry/).