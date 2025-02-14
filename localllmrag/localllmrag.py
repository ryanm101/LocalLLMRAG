import os
import multiprocessing
from langchain_ollama import OllamaLLM
from localllmrag.validate_config import get_config
from localllmrag.util import get_vector_db, load_index_metadata, save_index_metadata, logger, get_files_to_process
from localllmrag.processor import process_file

# --- Disable parallelism for Hugging Face tokenizers to avoid warning and Telemetry to keep things local ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

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
    config = get_config()

    metadata_path = config["global"]["index_metadata_file"]
    indexed_files = load_index_metadata(metadata_path)

    files_to_process = get_files_to_process(config, indexed_files)
    vector_db = get_vector_db(config)
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