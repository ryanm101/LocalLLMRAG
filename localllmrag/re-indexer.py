import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from localllmrag.validate_config import get_config
from localllmrag.util import load_index_metadata, save_index_metadata, get_vector_db, logger
from localllmrag.processor import process_file

# --- Disable parallelism for Hugging Face tokenizers to avoid warning and Telemetry to keep things local ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY_222"] = "False"


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
        # Skip files in excluded directories and check file types.
        if any(excluded in filepath.split(os.sep) for excluded in self.excluded_dirs) or not filepath.endswith(('.py', '.java', '.js', '.c', '.cpp', '.ts', '.cs')):
            return
        logger.info(f"Detected change in: {filepath}")
        self.reindex_file(filepath)

    def reindex_file(self, filepath):
        mod_time, file_hash, code_chunks = process_file(filepath)  # Assuming process_file is defined in utils or directly imported
        if code_chunks:
            self.vector_db.add_documents(code_chunks)
            self.indexed_files[filepath] = {"mod_time": mod_time, "hash": file_hash}
            save_index_metadata(self.indexed_files, self.metadata_path)
            logger.info(f"Re-indexed {filepath} with {len(code_chunks)} chunks.")
        else:
            logger.warning(f"No chunks produced for {filepath}.")

def main():
    config = get_config()
    vector_db = get_vector_db(config)  # Define load_vector_db if it includes specific configurations
    event_handler = CodeChangeHandler(vector_db)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
