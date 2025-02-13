import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Disable parallelism for Hugging Face tokenizers to avoid warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(process)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_vector_db():
    """
    Loads the persistent vector database using the same embeddings model and persist directory as in localllmrag.py.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    # Use the same persist directory as defined in localllmrag.py.
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
    return vector_db


def test_retrieval(vector_db, query="logger.info"):
    """
    Retrieves and logs a snippet of context from the vector database for the given query.
    Uses retriever.invoke(query) which is the current method.
    """
    retriever = vector_db.as_retriever()
    relevant_chunks = retriever.invoke(query)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)
    logger.info("Retrieved %d chunks for query: %s", len(relevant_chunks), query)
    logger.info("Context snippet:\n%s", context[:300])
    return context


def count_documents(vector_db):
    """
    Optionally, attempt to count the number of documents in the vector database.
    Note: This relies on internal API details of Chroma which may change.
    """
    try:
        # This is a hypothetical method. Check your Chroma API for the correct property/method.
        count = vector_db._collection.count()
        logger.info("Total documents in vector DB: %d", count)
        return count
    except Exception as e:
        logger.error("Could not retrieve document count: %s", e)
        return None


def main():
    vector_db = load_vector_db()
    # Optionally count documents (if supported by your Chroma version).
    count_documents(vector_db)

    # Prompt for a test query (default to "logger.info")
    query = input("Enter a test query for the vector DB (default 'logger.info'): ").strip() or "logger.info"
    context = test_retrieval(vector_db, query)

    print("Retrieved context snippet:")
    print(context[:500])


if __name__ == "__main__":
    main()
