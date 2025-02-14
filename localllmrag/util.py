import os
import yaml
import json
import hashlib
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(process)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_files_to_process(config, indexed_files):
    global_include_file_types = config["global"]["include_file_types"]
    global_exclude_dirs = config["global"]["exclude_dirs"]
    filepaths = []
    for dir_conf in config["dirs"]:
        directory = dir_conf["path"]
        include_types = dir_conf.get("include_file_types", global_include_file_types)
        exclude_dirs = dir_conf.get("exclude_dirs", global_exclude_dirs)
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if any(file.endswith(ext) for ext in include_types):
                    filepaths.append(os.path.join(root, file))
    logger.info(f"Found {len(filepaths)} files to consider for indexing.")

    files_to_process = []
    for filepath in filepaths:
        try:
            mod_time = os.path.getmtime(filepath)
            if filepath in indexed_files and indexed_files[filepath]["mod_time"] == mod_time:
                logger.info(f"Skipping {filepath}: already indexed and unchanged.")
                continue
            file_hash = compute_file_hash(filepath)
            files_to_process.append((filepath, mod_time, file_hash))
        except Exception as e:
            logger.error(f"Error processing metadata for {filepath}: {e}")

    logger.info(f"Found {len(files_to_process)} files to process.")

    return files_to_process

def get_vector_db(config):
    return Chroma(persist_directory=config["global"]["vector_db_dir"],
                       embedding_function=HuggingFaceEmbeddings(model_name=config["global"]["embeddings_model"]))

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_index_metadata(metadata_path="index_metadata.json"):
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}

def save_index_metadata(metadata, metadata_path="index_metadata.json"):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def get_language_for_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    from langchain.text_splitter import Language  # Import here to avoid circular imports if necessary
    mapping = {
        '.py': Language.PYTHON,
        '.java': Language.JAVA,
        '.js': Language.JS,
        '.ts': Language.JS,
        '.c': Language.C,
        '.cpp': Language.CPP,
        '.cs': Language.CSHARP,
        '.go': Language.GO
    }
    return mapping.get(ext, Language.PYTHON)

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_file_path = os.path.join(script_dir, 'config.schema.json')
    if not os.path.exists(schema_file_path):
        raise FileNotFoundError(f"Schema file '{schema_file_path}' does not exist.")

    with open(schema_file_path, "r") as f:
        schema = json.load(f)
    return schema