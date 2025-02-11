import os
import json
import hashlib
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Disable parallelism for Hugging Face tokenizers to avoid warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["Test"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(process)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Helper Functions for Metadata Management ---

def compute_file_hash(filepath):
    """Compute the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_index_metadata(metadata_path="index_metadata.json"):
    """Load file-index metadata from a JSON file if it exists."""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}

def save_index_metadata(metadata, metadata_path="index_metadata.json"):
    """Save the file-index metadata to a JSON file."""
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

# --- Code Processing Functions (adapted from rag.py) ---

def get_language_for_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        '.py': Language.PYTHON,
        '.java': Language.JAVA,
        '.js': Language.JS,
        '.ts': Language.JS,
        '.c': Language.C,
        '.cpp': Language.CPP,
        '.cs': Language.CSHARP
    }
    return mapping.get(ext, Language.PYTHON)

def process_file(filepath):
    """
    Processes a single file to load, chunk, and return the results.
    Returns a tuple: (mod_time, file_hash, code_chunks)
    """
    try:
        logger.info(f"Processing file for re-indexing: {filepath}")
        mod_time = os.path.getmtime(filepath)
        file_hash = compute_file_hash(filepath)

        loader = TextLoader(filepath)
        code_document = loader.load()
        language = get_language_for_file(filepath)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=1000, chunk_overlap=100
        )
        code_chunks = splitter.split_documents(code_document)

        # Prepend file name to each chunk's content
        for chunk in code_chunks:
            chunk.page_content = f"File: {filepath}\n{chunk.page_content}"

        logger.info(f"{filepath} produced {len(code_chunks)} chunks.")
        return mod_time, file_hash, code_chunks
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None, None, []

# --- Vector DB Loader ---

def load_vector_db():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    # Use the same persist directory as in rag.py.
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
    return vector_db

# --- Watchdog Event Handler ---

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, vector_db, metadata_path="index_metadata.json"):
        self.vector_db = vector_db
        self.metadata_path = metadata_path
        self.indexed_files = load_index_metadata(metadata_path)
        self.excluded_dirs = {"venv", ".venv", "node_modules"}

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_event(event.src_path)

    def _handle_event(self, filepath):
        # Process only specific file types.
        if not filepath.endswith(('.py', '.java', '.js', '.c', '.cpp', '.ts', '.cs')):
            return
        # Skip files in excluded directories.
        for excluded in self.excluded_dirs:
            if excluded in filepath.split(os.sep):
                return
        logger.info(f"Detected change in: {filepath}")
        self.reindex_file(filepath)

    def reindex_file(self, filepath):
        try:
            mod_time_new = os.path.getmtime(filepath)
            # Skip re-indexing if file hasn't changed.
            if filepath in self.indexed_files and self.indexed_files[filepath]["mod_time"] == mod_time_new:
                logger.info(f"{filepath} unchanged. Skipping re-indexing.")
                return

            mod_time, file_hash, code_chunks = process_file(filepath)
            if code_chunks:
                # Add new chunks to the vector DB.
                self.vector_db.add_documents(code_chunks)
                # Update metadata.
                self.indexed_files[filepath] = {"mod_time": mod_time, "hash": file_hash}
                save_index_metadata(self.indexed_files, self.metadata_path)
                logger.info(f"Re-indexed {filepath} with {len(code_chunks)} chunks.")
            else:
                logger.warning(f"No chunks produced for {filepath}.")
        except Exception as e:
            logger.error(f"Error re-indexing {filepath}: {e}")

# --- Main Function to Run the Watcher ---

def main():
    vector_db = load_vector_db()
    event_handler = CodeChangeHandler(vector_db)
    observer = Observer()
    # Monitor the current directory recursively.
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()
    logger.info("Background re-indexer started. Monitoring for file changes...")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
