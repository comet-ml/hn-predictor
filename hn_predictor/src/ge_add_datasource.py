import great_expectations as ge
from ruamel import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='hn_datasource', type=str)
    parser.add_argument('--base_directory', default='../data/', type=str)
    return parser.parse_args()


def create_datasource(context, name, base_directory):
    # First configure a new Datasource and add to DataContext
    datasource_yaml = f"""
    name: {name}
    class_name: Datasource
    module_name: great_expectations.datasource
    execution_engine:
        module_name: great_expectations.execution_engine
        class_name: PandasExecutionEngine
    data_connectors:
        default_runtime_data_connector_name:
            class_name: RuntimeDataConnector
            batch_identifiers:
                - default_identifier_name
        default_inferred_data_connector_name:
            class_name: InferredAssetFilesystemDataConnector
            base_directory: {base_directory}
            default_regex:
                group_names:
                    - data_asset_name
                pattern: (.*)
    """

    context.test_yaml_config(datasource_yaml)
    context.add_datasource(**yaml.load(datasource_yaml))


if __name__ == '__main__':
    args = get_args()
    context = ge.get_context()
    create_datasource(context, args.name, args.base_directory)
