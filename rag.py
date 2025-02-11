import os
import json
import hashlib
import logging
import multiprocessing
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# Disable parallelism for Hugging Face tokenizers to avoid warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

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

# --- Worker Functions for Processing Files ---
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
    return mapping.get(ext, Language.PYTHON)  # Default to Python if not found

def process_file(args):
    """
    Processes a single file to load, chunk, and return the results.
    Expects args as a tuple: (filepath, mod_time, file_hash)
    Returns a tuple: (filepath, mod_time, file_hash, code_chunks)
    """
    filepath, mod_time, file_hash = args
    logger.info(f"Process ID: {os.getpid()} - Processing: {filepath}")

    try:
        loader = TextLoader(filepath)
        code_document = loader.load()

        # Dynamically select the language based on the file extension.
        language = get_language_for_file(filepath)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=1000, chunk_overlap=100
        )
        code_chunks = splitter.split_documents(code_document)

        # Prepend the file name to each chunk's content.
        for chunk in code_chunks:
            chunk.page_content = f"File: {filepath}\n{chunk.page_content}"

        logger.info(f"Process ID: {os.getpid()} - {filepath} produced {len(code_chunks)} chunks.")
        return filepath, mod_time, file_hash, code_chunks
    except Exception as e:
        logger.error(f"Process ID: {os.getpid()} - Error processing {filepath}: {e}")
        return filepath, mod_time, file_hash, []


# --- Functions for Retrieval and RAG Answering ---

def retrieve_context(vector_db, query):
    """Retrieves relevant code chunks from the vector database."""
    retriever = vector_db.as_retriever()
    relevant_chunks = retriever.invoke(query)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    logger.debug(f"Retrieved {len(relevant_chunks)} chunks for query: {query}")
    logger.debug(f"Context snippet: {context[:200]}")
    return context

def rag_answer(vector_db, ollama_llm, query):
    """Generates an answer using the RAG pipeline with Ollama."""
    context = retrieve_context(vector_db, query)
    augmented_prompt = (
        "Use the following code context to answer the question at the end. "
        "If you cannot answer, just say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    answer = ollama_llm.invoke(augmented_prompt)
    return answer

# --- Main Process ---

if __name__ == "__main__":
    code_directory = "./"  # Adjust as needed.
    excluded_dirs = {"venv", ".venv", "node_modules"}
    filepaths = []
    for root, dirs, files in os.walk(code_directory):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if file.endswith(('.py', '.java', '.js', '.c', '.cpp')):
                filepaths.append(os.path.join(root, file))
    logger.info(f"Found {len(filepaths)} files to consider for indexing.")

    metadata_path = "index_metadata.json"
    indexed_files = load_index_metadata(metadata_path)
    files_to_process = []

    for filepath in filepaths:
        try:
            mod_time = os.path.getmtime(filepath)
            # If the file exists in metadata and the modification time hasn’t changed, skip it.
            if filepath in indexed_files and indexed_files[filepath]["mod_time"] == mod_time:
                logger.info(f"Skipping {filepath}: already indexed and unchanged.")
                continue
            # Otherwise, compute the file hash.
            file_hash = compute_file_hash(filepath)
            files_to_process.append((filepath, mod_time, file_hash))
        except Exception as e:
            logger.error(f"Error processing metadata for {filepath}: {e}")

    logger.info(f"Found {len(files_to_process)} files to process.")

    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
    ollama_llm = OllamaLLM(model="llama3.1")

    batch_size = 10  # Adjust based on available memory and project size.
    num_processors = multiprocessing.cpu_count()
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} files.")
        with multiprocessing.Pool(processes=num_processors - 1 if num_processors > 1 else 1) as pool:
            results = pool.map(process_file, batch)
        batch_chunks = []
        for filepath, mod_time, file_hash, chunks in results:
            if chunks:
                batch_chunks.extend(chunks)
                indexed_files[filepath] = {"mod_time": mod_time, "hash": file_hash}
        vector_db.add_documents(batch_chunks)
        logger.debug("Sample chunk from indexed data: %s", batch_chunks[0].page_content if batch_chunks else "No chunks")
        logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch_chunks)} chunks added.")
        save_index_metadata(indexed_files, metadata_path)

    logger.info(f"Indexing complete. RAG system ready for queries.")

    while True:
        user_query = input("Ask a question about the code (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = rag_answer(vector_db, ollama_llm, user_query)
        print(f"\nQuestion: {user_query}\n")
        print(f"Answer: {answer}\n")
        print("-" * 50 + "\n")