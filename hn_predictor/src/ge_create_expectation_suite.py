import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.render.renderer import ValidationResultsPageRenderer
from great_expectations.render.view import DefaultJinjaPageView
from ruamel import yaml

from ge_add_datasource import create_datasource

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

context = ge.get_context()

expectation_suite_name = "hn_suite.demo"
# Create a new expectation suite
expectation_suite = context.create_expectation_suite(
    expectation_suite_name, overwrite_existing=True
)
batch_request = BatchRequest(
    datasource_name="hn_datasource",
    data_connector_name="default_inferred_data_connector_name",
    data_asset_name="train.pkl",
    limit=1000,
)
validator = context.get_validator(
    batch_request=batch_request, expectation_suite_name=expectation_suite_name
)

validator.validate()

# Declare the expectatios for the data here
def create_expectations(ge_df, columns=None):
    ge_df.expect_column_values_to_be_in_type_list("score", ["int", "float"])
    ge_df.expect_column_values_to_not_be_null("score", mostly=0.95)
    ge_df.expect_column_median_to_be_between("score", 1, 3)

    # This will always fail
    ge_df.expect_column_mean_to_be_between("score", 1, 2)


create_expectations(validator, FEATURE_COLUMNS)
validator.save_expectation_suite(discard_failed_expectations=False)
