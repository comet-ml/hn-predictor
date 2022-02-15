import argparse
import os

import comet_ml
import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.render.renderer import ValidationResultsPageRenderer
from great_expectations.render.view import DefaultJinjaPageView
from ruamel import yaml

from data import fetch_dataset_artifact, load_data
from ge_create_expectation_suite import create_expectations

DATA_PATH = os.getenv("DATA_PATH", "../data")

FEATURE_COLUMNS = [
    "id",
    "title",
    "url",
    "text",
    "dead",
    "by",
    "score",
    "time",
    "timestamp",
    "type",
    "parent",
    "descendants",
    "ranking",
    "deleted",
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_name", default='hn-dataset', type=str)
    parser.add_argument("--artifact_version", default='1.0.0', type=str)
    parser.add_argument("--split_name", default='valid', type=str)
    parser.add_argument("--expectation_suite_name", default='hn_suite.demo', type=str)
    parser.add_argument("--validation_threshold", default=75, type=int)
    parser.add_argument("--message", default='validate suite', type=str)
    return parser.parse_args()

def create_validation_report(batch):
    validation_result = batch.validate()
    document_model = ValidationResultsPageRenderer().render(validation_result)
    report = DefaultJinjaPageView().render(document_model)
    return validation_result, report

if __name__ == "__main__":
    args = get_args()
    context = ge.get_context()
    experiment = comet_ml.Experiment()


    data_config = fetch_dataset_artifact(
        experiment,
        artifact_name=args.artifact_name,
        artifact_version=args.artifact_version,
        output_path=DATA_PATH,
    )
    
    data_path = data_config[args.split_name]
    print(data_path)

    expectation_suite_name = args.expectation_suite_name
    batch_request = BatchRequest(
        datasource_name="hn_datasource",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name=os.path.basename(data_path),
        limit=1000,
    )
    validator = context.get_validator(
        batch_request=batch_request, expectation_suite_name=expectation_suite_name
    )

    create_expectations(validator, FEATURE_COLUMNS)
    validator.save_expectation_suite(discard_failed_expectations=False)

    threshold = 70

    name1, name2 = expectation_suite_name.split(".")
    experiment.log_asset(
        f"../great_expectations/expectations/{name1}/{name2}.json",
        f"{expectation_suite_name}.json",
    )
    results, report = create_validation_report(validator)

    # Log the validation threshold that determines whether tests have passed on failed
    experiment.log_parameter("validation_threshold", threshold)

    # Log the name of the expectation suite being used to validate the data.
    experiment.log_parameter("expectation_suite_name", expectation_suite_name)

    # Log the generated html validation report
    experiment.log_html(report)

    # Log the summary statistics of the validation
    experiment.log_metrics(results.statistics)
    if results.statistics["success_percent"] < threshold:
        experiment.add_tag("failed")
    else:
        experiment.add_tag("success")

    experiment.end()
