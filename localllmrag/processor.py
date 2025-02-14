import os

import ast  # For Python
import javalang # For Java
import jsbeautifier # For JS and TS
import clang # For C and CPP
import astor # For C#
import goastpy # For GO

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from localllmrag.util import logger, get_language_for_file

def load_and_get_text(filepath):
    """Loads a file and returns its text content."""
    try:
        loader = TextLoader(filepath)
        code_document = loader.load()
        return code_document[0].page_content
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        return None

def perform_semantic_chunking(text, language, filepath):
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

def get_code_chunks(text, language, filepath):
    code_chunks = perform_semantic_chunking(text, language, filepath)
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
    for chunk_content in get_code_chunks(text, language, filepath):  # Iterate over the strings
        code_chunks_as_documents.append(Document(page_content=f"File: {filepath}\n{chunk_content}"))  # Add the Document to the list

    logger.info(f"Process ID: {os.getpid()} - {filepath} produced {len(code_chunks_as_documents)} chunks.") # Log the amount of documents
    return filepath, mod_time, file_hash, code_chunks_as_documents  # Return the list of Documents