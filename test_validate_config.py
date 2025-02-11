import pytest
import jsonschema
from util import load_schema
from validate_config import validate_config

# A sample valid configuration.
VALID_CONFIG = {
    "global": {
        "include_file_types": [".py", ".js"],
        "exclude_dirs": ["venv", ".venv", "node_modules"]
    },
    "dirs": [
        {
            "path": "./src",
            "include_file_types": [".py"],
            "exclude_dirs": ["tests", "__pycache__"]
        }
    ]
}

# A sample invalid configuration (e.g., missing the required 'exclude_dirs' in the global section).
INVALID_CONFIG = {
    "global": {
        "include_file_types": [".py", ".js"]
        # Missing 'exclude_dirs'
    },
    "dirs": []
}


def test_valid_config(tmp_path):
    schema = load_schema(schema_path="./config.schema.json")
    validate_config(VALID_CONFIG, schema)


def test_invalid_config(tmp_path):
    schema = load_schema(schema_path="./config.schema.json")

    # Expect a ValidationError due to the missing required property.
    with pytest.raises(jsonschema.ValidationError):
        validate_config(INVALID_CONFIG, schema)