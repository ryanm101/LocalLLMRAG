import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag import load_and_index_code_file # Import the indexing function from rag_code_qa.py

class CodeFileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return None
        if event.event_type == 'modified':
            filepath = event.src_path
            if filepath.endswith(('.py', '.java', '.js', '.c', '.cpp')):
                print(f"Code file modified: {filepath}")
                load_and_index_code_file(filepath) # Call the indexing function

if __name__ == "__main__":
    path_to_watch = "./code_files" # Directory to monitor

    event_handler = CodeFileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    print(f"Watching for code file changes in: {path_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()