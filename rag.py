import os
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM  # Import for Ollama integration

embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vector_db = Chroma(persist_directory="./chroma_db_code", embedding_function=embeddings_model)

ollama_llm = OllamaLLM(model="llama3.1")

def load_and_index_code_file(filepath):
    """Loads a single code file, chunks it, embeds, and adds to vector DB."""
    print(f"Loading and indexing: {filepath}")
    try:
        loader = TextLoader(filepath) # Generic TextLoader - could be replaced with CodeLoader for more structure
        code_document = loader.load()
        print(f"Loaded document: {filepath}")

        # Code-aware splitter (Python example - adapt Language enum if needed)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=100)
        code_chunks = python_splitter.split_documents(code_document)
        print(f"Split into {len(code_chunks)} chunks.")


        vector_db.add_documents(code_chunks) # Add to vector DB
        print(f"Indexed {filepath} - {len(code_chunks)} chunks.")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def retrieve_context(query):
    """Retrieves relevant code chunks from vector DB."""
    retriever = vector_db.as_retriever()
    relevant_chunks = retriever.invoke(query)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

def rag_answer(query):
    """Generates answer using RAG pipeline with Ollama."""
    context = retrieve_context(query)
    augmented_prompt = f"Use the following code context to answer the question at the end. If you cannot answer, just say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"
    answer = ollama_llm.invoke(augmented_prompt) # Use Ollama LLM for generation
    return answer

if __name__ == "__main__":
    code_directory = "./code_files" # Directory containing code files

    print("Initial indexing of code directory...")
    for root, _, files in os.walk(code_directory):
        for file in files:
            if file.endswith(('.py', '.java', '.js', '.c', '.cpp')):
                filepath = os.path.join(root, file)
                load_and_index_code_file(filepath)
    print("Initial indexing complete. RAG system ready for queries.")

    while True:
        user_query = input("Ask a question about the code (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = rag_answer(user_query)
        print(f"Question: {user_query}")
        print(f"Answer: {answer}")
        print("\n" + "-"*50 + "\n")