import os
import json
import hashlib
import logging
import multiprocessing

import ast  # For Python
import javalang # For Java
import jsbeautifier # For JS and TS
import clang # For C and CPP
import astor # For C#
import goastpy # For GO

from jsonschema.exceptions import ValidationError
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from localllmrag.validate_config import validate_config
from localllmrag.util import load_schema, load_config

# --- Disable parallelism for Hugging Face tokenizers to avoid warning and Telemetry to keep things local ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- Setup Logging ---
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
        '.cs': Language.CSHARP,
        '.md': Language.MARKDOWN,
        '.go': Language.GO,
    }
    return mapping.get(ext, Language.PYTHON)  # Default to Python if not found

def load_and_get_text(filepath):
    """Loads a file and returns its text content."""
    try:
        loader = TextLoader(filepath)
        code_document = loader.load()
        return code_document[0].page_content
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        return None

def perform_semantic_chunking(text, language):
    """Performs semantic chunking based on the language."""
    code_chunks = []
    if language in [Language.PYTHON, Language.JAVA, Language.JS, Language.TS, Language.C, Language.CPP, Language.CSHARP, Language.GO]:
        try:

            if language == Language.PYTHON:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        chunk_content = ast.unparse(node)
                        if chunk_content:
                            code_chunks.append(f"{chunk_content}")
            elif language == Language.JAVA:
                tree = javalang.parse.parse(text)
                for node in tree.types:
                  if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                    chunk_content = node
                    if chunk_content: #check for empty chunks
                      code_chunks.append(f"{chunk_content}")
            elif language == Language.JS or language == Language.TS:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                opts.space_before_conditional = True
                tree = ast.parse(jsbeautifier.beautify(text, opts))
                for node in ast.walk(tree):
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk_content = ast.unparse(node)
                    if chunk_content: #check for empty chunks
                      code_chunks.append(f"{chunk_content}")
            elif language == Language.C or language == Language.CPP:
              index = clang.cindex.Index.create()
              tu = index.parse(filepath)
              for node in tu.cursor.get_children():
                if node.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CLASS_DECL]:
                    extent = node.extent
                    chunk_content = text[extent.start.offset:extent.end.offset]
                    if chunk_content: #check for empty chunks
                      code_chunks.append(f"{chunk_content}")
            elif language == Language.CSHARP:
              tree = astor.parse_file(filepath)
              for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                  chunk_content = astor.to_source(node)
                  if chunk_content: #check for empty chunks
                    code_chunks.append(f"{chunk_content}")
            elif language == Language.GO: # TODO: This has not been tested yet....
              tree = goastpy.GoAst(filepath)
              for node in ast.walk(tree):
                if isinstance(node, (ast.FuncDecl, ast.TypeSpec)):
                  chunk_content = goastpy.GoAst.parse_source_code_to_json(node)
                  if chunk_content: #check for empty chunks
                    code_chunks.append(f"{chunk_content}")

        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}")  # More general message
            return []  # Return empty list if semantic chunking fails

    return code_chunks


def perform_recursive_chunking(text, language, chunk_size=1500, chunk_overlap=150):
    """Performs recursive chunking."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(text)
    return [doc.page_content for doc in docs] # Return only page_content

def get_code_chunks(text, language):
    code_chunks = perform_semantic_chunking(text, language)
    if not code_chunks:
        code_chunks = perform_recursive_chunking(text, language)
    return code_chunks

def process_file(args):
    """Processes a single file to load, chunk, and return the results."""
    filepath, mod_time, file_hash = args
    logger.info(f"Process ID: {os.getpid()} - Processing: {filepath}")

    text = load_and_get_text(filepath)
    if text is None:  # Handle file loading errors
        return filepath, mod_time, file_hash, []

    language = get_language_for_file(filepath)

    code_chunks_as_documents = []  # New list to hold Document objects
    for chunk_content in get_code_chunks(text, language):  # Iterate over the strings
        code_chunks_as_documents.append(Document(page_content=f"File: {filepath}\n{chunk_content}"))  # Add the Document to the list

    logger.info(f"Process ID: {os.getpid()} - {filepath} produced {len(code_chunks_as_documents)} chunks.") # Log the amount of documents
    return filepath, mod_time, file_hash, code_chunks_as_documents  # Return the list of Documents

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
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Environment: Code review and improvement\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Context:\n{context}\n\n"
        "Instruction: Using the code context above, review and improve the code as needed. "
        "When suggesting code changes, please provide your answer using markdown code blocks. "
        "If the context does not provide enough information, simply respond with 'I don't know'.\n"
        f"Question: {query}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return ollama_llm.invoke(augmented_prompt)

# --- Main Process ---
if __name__ == "__main__":
    config = load_config()
    schema = load_schema()
    if not validate_config(config, schema):
        logger.error("Config validation failed")
        raise ValidationError("Config validation failed")

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

    metadata_path = config["global"]["index_metadata_file"]
    indexed_files = load_index_metadata(metadata_path)
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

    embeddings_model = HuggingFaceEmbeddings(model_name=config["global"]["embeddings_model"])
    vector_db = Chroma(persist_directory=config["global"]["vector_db_dir"], embedding_function=embeddings_model)
    ollama_llm = OllamaLLM(model=config["global"]["llm_model"])

    batch_size = 10  # Adjust based on available memory and project size.
    num_processors = multiprocessing.cpu_count()
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} files.")
        with multiprocessing.Pool(processes=num_processors - 1 if num_processors > 1 else 1) as pool:
            results = pool.map(process_file, batch)
        batch_chunks = []
        for filepath, mod_time, file_hash, chunks in results:  # chunks is now a list of Documents
            if chunks:
                batch_chunks.extend(chunks)  # Extend with Documents
            indexed_files[filepath] = {"mod_time": mod_time, "hash": file_hash}
        if len(batch_chunks) > 0:
            vector_db.add_documents(batch_chunks)  # Now adding Documents
        logger.debug("Sample chunk from indexed data: %s", batch_chunks[0].page_content if batch_chunks else "No chunks")
        logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch_chunks)} chunks added.")
        save_index_metadata(indexed_files, metadata_path)

    logger.info("Indexing complete. RAG system ready for queries.")

    while True:
        user_query = input("Ask a question about the code (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = rag_answer(vector_db, ollama_llm, user_query)
        print(f"\nQuestion: {user_query}\n")
        print(f"Answer: {answer}\n")
        print("-" * 50 + "\n")