# hn-predictor

### Experimentation Requirements

1. When running an experiment, pass in a message (similar to a git commit message) using the command line args in order to keep the project organized
2. When logging a prediction experiment, save the predictions as a `predictions.csv` and log them as an experiment asset
3. Log trained model assets using `experiment.log_model()` so that they can be added to the [Model Registry](https://www.comet.ml/site/using-comet-model-registry/).