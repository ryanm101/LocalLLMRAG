import os
import yaml
import json

def load_config(config_path="config.yaml"):
    """
    Loads the configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' does not exist.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_schema(schema_path="config.schema.json"):
    """
    Loads the JSON schema from a file.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file '{schema_path}' does not exist.")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return schema