## Local RAG for Code with Automatic Updates (Ollama Version)
This example demonstrates how to build a Retrieval-Augmented Generation (RAG) system that reads local code source files and automatically updates when those files are modified, using Ollama as the local LLM for question answering.

```
Directory Structure
/
├── chroma_db_code/         # Directory for ChromaDB vector database (persistent) - Not packaged
├── code_files/             # Directory to store example code files - Not packaged
│   ├── example1.py
│   └── example2.py
├── tests/                  # Directory to store example code files
│   ├── verify_testing.py
│   └── test_validate_config.py
├── localllmrag/            # Application Directory
│   ├── __init__.py   
│   ├── config.schema.json  # Schema for Config file   
│   ├── localllmrag.py      # Python script for RAG pipeline and initial indexing (Ollama version)
│   ├── re-indexer.py       # Python script for file monitoring and update triggering
│   ├── util.py             # Helper functions to load schema and config
│   └── validate_config.py  # Verifies the config against the schema
├── project.toml
├── Makefile
├── requirements.txt
├── MANIFEST.in
├── config.yaml.sample
└── README.md
```

---

### Setup Instructions

1. **Install Ollama:**

   - Download and install Ollama for your operating system from the official Ollama website. Follow their installation instructions.
   - Pull an LLM Model: Once Ollama is installed and running, open your terminal and run a command like:
     ```bash
     ollama pull llama3.1
     ```
     Replace `llama3.1` with the name of the model you prefer (e.g., `mistral`, `codellama`). Make sure the model name you pull here matches the one specified in your `config.yaml`.

2. **Install Python Libraries:**

   - Ensure you have Python 3.8 or later installed.
   - Install the required Python libraries using pip:
     ```bash
     pip install -r requirements.txt
     ```
     (The `requirements.txt` file should include dependencies such as `langchain-ollama`, `pyyaml`, `jsonschema`, etc.)

3. **Configure Your System via `config.yaml`:**

   - Open the `config.yaml` file in your favorite text editor. This file lets you customize:
     - **Global Settings:**  
       File types to include, directories to exclude, the name of the index metadata file, the ChromaDB directory, the Ollama LLM model, and the embeddings model.
     - **Directory-Specific Settings:**  
       Override global defaults on a per-directory basis.
       
   - Example `config.yaml`:
     ```yaml
     global:
       # Default file types to include when scanning directories.
       include_file_types:
         - .py
         - .java
         - .js
         - .c
         - .cpp
         - .ts
         - .cs
         - .go
         - .md

       # Default directories to exclude from scanning.
       exclude_dirs:
         - venv
         - .venv
         - node_modules
         - __pycache__

       index_metadata_file: "index_metadata.json"
       vector_db_dir: "./chroma_db_code"
       llm_model: "llama3.1"
       embeddings_model: "all-mpnet-base-v2"

     dirs:
       # Example directory configuration for the root directory.
       - path: "./"
         # Override to scan only Python files in this directory.
         include_file_types:
           - .py
         exclude_dirs:
           - .go

       # Additional directory configurations can be added below:
       - path: "./xxx"
         exclude_dirs:
           - .go

       - path: "./yyyy"
         include_file_types:
           - .py
     ```
   - The settings in `config.yaml` will be read by `rag.py` and used to control which files are indexed and which models/settings are used.

4. **Run Initial Indexing:**

   - In a terminal, navigate to the `rag/` directory.
   - Run the `rag.py` script to perform initial indexing and create the persistent vector database:
     ```bash
     python localllmrag.py
     ```
   - You should see output indicating that the code files are being loaded and indexed. Wait until you see a message like "Indexing complete. RAG system ready for queries." before proceeding.

5. **Run the File Monitor (for Automatic Updates):**

   - Open a new terminal window and navigate to the `rag/` directory.
   - Start the re-indexer script (which monitors for file changes and updates the index automatically):
     ```bash
     python re-indexer.py
     ```
   - This script will continuously watch the directories specified in `config.yaml` for modifications and re-index changed files on the fly.

---

### How to Use and Test

1. **Ask Questions (Interactive Querying):**

   - With `rag.py` running, you can ask questions about your code directly in the terminal. For example:
     ```bash
     Ask a question about the code (or type 'exit' to quit): What does the calculate_sum function do?
     ```
   - The RAG system will use the indexed code context and Ollama to answer based on your local code files.
   - Type `exit` to end the interactive session.

2. **Modify Code Files and Observe Automatic Updates:**

   - While `re-indexer.py` is running, open one of the code files in `code_files/` (or any directory specified in `config.yaml`).
   - Make a change (e.g., update a comment or modify a function) and save the file.
   - Check the terminal where `re-indexer.py` is running: it should log that the file was modified and re-indexed.
   - Return to the interactive session in `rag.py` and ask a question that reflects the change. The answer should now include the updated content.

---

### Important Notes

- **Ollama Setup:**  
  Ensure Ollama is installed, running, and that the model specified in your `config.yaml` is pulled and available. If you encounter issues, verify that the Ollama server is active (you may need to run `ollama serve` in a separate terminal).

- **Resource Usage:**  
  Running local LLMs and embedding models can be resource-intensive. Monitor your system’s CPU, RAM, and GPU usage—especially when working with large codebases or high-resource models. Adjust chunk sizes, directory scopes, and model settings if necessary.

- **ChromaDB Persistence:**  
  The vector database is stored persistently in the directory specified by `vector_db_dir` in `config.yaml`. If you delete this directory, you will need to re-run `rag.py` for initial indexing.

- **Error Handling:**  
  The scripts include basic error handling. For production use or more complex environments, consider enhancing the error handling and logging as needed.

- **Customizing for Different Code Languages:**  
  If you work with multiple programming languages, update the file extension mappings in `get_language_for_file()` (in `rag.py`) accordingly.

---

With these updates, your RAG system is now fully configurable via `config.yaml`, making it easier to adapt to different projects and environments without modifying code.

Happy coding and querying!
