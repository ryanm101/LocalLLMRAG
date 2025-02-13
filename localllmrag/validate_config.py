from jsonschema import validate, ValidationError
from .util import load_config, load_schema

def validate_config(config, schema):
    """
    Validates the configuration against the provided schema.
    Raises a jsonschema.ValidationError if the config is invalid.
    """
    try:
        validate(instance=config, schema=schema)
        return True, None
    except ValidationError as e:
        error_info = {
            "message": e.message,
            "path": list(e.path),  # Convert deque to list for easier handling
            "schema_path": list(e.schema_path),
            "instance": e.instance,
            "schema": e.schema
        }
        return False, error_info

if __name__ == "__main__":
    try:
        config = load_config("config.yaml")
        schema = load_schema("config.schema.json")
        validate_config(config, schema)
        print("Configuration is valid!")
    except ValidationError as e:
        print("Configuration is invalid!")
        print(e)
    except Exception as ex:
        print("Error loading configuration or schema:")
        print(ex)
