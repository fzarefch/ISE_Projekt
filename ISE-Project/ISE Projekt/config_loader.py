import yaml


class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.yaml_config = self.load_yaml()

    def load_yaml(self):
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get_database_config(self):
        return self.yaml_config.get("database", {})

    def get_csv_config(self):
        return self.yaml_config.get("csv_file", {})

    def get_queries(self):
        return self.yaml_config.get("queries", {})