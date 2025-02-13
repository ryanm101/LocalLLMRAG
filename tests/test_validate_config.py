import pytest
from jsonschema import ValidationError
from localllmrag.util import load_schema
from localllmrag.validate_config import validate_config

# A sample valid configuration.
VALID_CONFIG = {
    "global": {
        "include_file_types": [".py", ".js"],
        "exclude_dirs": ["venv", ".venv", "node_modules"],
        "index_metadata_file": "index_metadata.json",
        "vector_db_dir": "./chroma_db",
        "llm_model": "llama3.1",
        "embeddings_model": "all-mpnet-base-v2",
        "chunk_size": 1500,
        "chunk_overlap": 150
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
    res, e = validate_config(VALID_CONFIG, schema)
    if e is not None:
        print(e)
    assert res is True
    assert e is None

def test_invalid_config(tmp_path):
    schema = load_schema(schema_path="./config.schema.json")
    res, e = validate_config(INVALID_CONFIG, schema)
    assert res is False
    assert e is not None