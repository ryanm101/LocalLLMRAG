## Local RAG for Code with Automatic Updates (Ollama Version)
This example demonstrates how to build a Retrieval-Augmented Generation (RAG) system that reads local code source files and automatically updates when those files are modified, using Ollama as the local LLM for question answering.

```
Directory Structure
rag/
├── code_files/         # Directory to store example code files
│   ├── example1.py
│   └── example2.py
├── chroma_db_code/     # Directory for ChromaDB vector database (persistent)
├── rag.py              # Python script for RAG pipeline and initial indexing (Ollama version)
├── monitor.py          # Python script for file monitoring and update triggering
└── README.md           # Instructions (this file)
```
### Setup Instructions
1. Install Ollama:

    Download and install Ollama for your operating system from the official Ollama website. Follow their installation instructions.
    Pull an LLM Model: Once Ollama is installed and running, you need to pull a model. Open your terminal and run a command like this to download the llama2 model (or choose another model from the Ollama model library):
    ```Bash
    
    ollama pull llama3.1
    ```
    Replace llama3.1 with the name of the model you prefer (e.g., mistral, codellama). Make sure the model name you pull here matches the one you configure in rag_code_qa.py in the next step.

2. Install Python Libraries:

    Make sure you have Python installed (preferably Python 3.8 or later). Then, install the required libraries using pip, including langchain-ollama:

    ```Bash
    pip install -r requirements.txt
    ```
   
3. Configure Ollama Model in rag.py:

    Open the rag_code_qa.py file in a text editor.
    Find the line that initializes the LLM generator:
    ```Python
    generator = OllamaLLM(model="llama3.1") # Example: Using the 'llama2' model. Change this to your desired model.
    ```
    Replace "llama2" with the name of the Ollama model you downloaded in the previous step (e.g., if you pulled mistral, change it to model="mistral").

4. Run Initial Indexing:

    Run the rag_code_qa.py script to perform the initial indexing of your code files and create the vector database. Open a terminal, navigate to the rag_code_example/ directory, and run:

    ```Bash
    python rag.py
    ```
    You should see output indicating that the code files are being loaded and indexed. Wait until you see "Initial indexing complete. RAG system ready for queries." before proceeding.

5. Run the File Monitor:

    To enable automatic updates when code files change, run the code_monitor.py script in a separate terminal window (still in the rag_code_example/ directory):
    ```Bash
    python monitor.py
    ```
    This script will start watching the code_files/ directory for modifications. Keep this script running in the background.

### How to Use and Test

1. Ask Questions (using rag_code_qa.py):

    With rag_code_qa.py script running (after initial indexing), you can ask questions about your code directly in the terminal. For example:
    
    ```Bash
    Ask a question about the code (or type 'exit' to quit): What does calculate_sum function do?
    ```
    The RAG system will use Ollama to answer based on the content of your code files. Type exit to quit the interactive query session.

2. Modify Code Files and Observe Updates:
   * While code_monitor.py is running in a separate terminal, open any of the Python files in the code_files/ directory (e.g., `code_files/example1.py`) using a text editor. 
   * Make a small change to the code, such as updating a comment, modifying a docstring, or altering the code itself. Save the file.
   * In the terminal window where `monitor.py` is running, you should see a message like "Code file modified: ./code_files/example1.py" followed by output indicating that the file is being re-indexed. This confirms that the file monitoring and automatic update mechanism is working.
   * Now, go back to the terminal window where rag_code_qa.py is running and ask a question that relates to the change you just made. The RAG system should now be aware of the updated code content and provide answers reflecting the modifications.

### Important Notes
   * Ollama Setup: Ensure you have Ollama installed and running before you run `rag.py` or `monitor.py`. Ollama needs to be serving the model in the background for the Langchain integration to work. If you encounter issues, double-check that the Ollama server is started (ollama serve in a separate terminal if needed).
   * Ollama Model Choice: The quality of the generated answers will heavily depend on the Ollama model you choose. Larger models are generally more capable but require more resources (RAM, VRAM). Experiment with different models available in the Ollama model library to find one that suits your needs and hardware. Models like llama2, mistral, and codellama are popular choices.
   * Resource Usage: Running local LLMs (like those in Ollama) and embedding models can be resource-intensive. Monitor your CPU, RAM, and GPU usage, especially when working with larger codebases or more powerful models. Adjust chunk sizes and model choices if you experience performance issues or run out of memory.
   * ChromaDB Persistence: The vector database (ChromaDB) is stored persistently in the chroma_db_code/ directory within your rag_code_example/ folder. This means that the indexed embeddings are saved to disk. If you delete the chroma_db_code/ directory, you will need to re-run rag_code_qa.py to perform the initial indexing again.
   * Error Handling: The provided scripts include basic error handling, but you may want to add more robust error handling for production or more complex scenarios.
   * Code Language Support: The code splitting in rag_code_qa.py is currently configured for Python code files (Language.PYTHON). If you intend to use this RAG system with code in other programming languages, you may need to adjust the language parameter in the RecursiveCharacterTextSplitter.from_language() initialization within the load_and_index_code_file function. Refer to the Langchain documentation for supported languages.