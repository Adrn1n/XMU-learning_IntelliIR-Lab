"""
Global configuration settings for the Information Retrieval System.

This module contains all configuration parameters including:
- Document collection settings (directory, file extensions, encoding)
- Logging configuration (format, level, rotation, output destinations)
- Query processing settings (cache size, input source, output options)
- Ollama model integration settings (system prompts for query generation and answer generation)
"""

import os

# Global settings for the information retrieval system

# Document collection settings
DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../data"
)  # Directory containing document collection
DOC_EXTENSIONS = [
    ""
]  # File extensions to process (empty string for files without extension)
DOC_EXTENSIONS_EXCLUDE = None  # File extensions to exclude, None for no exclusions
DOC_ENCODING = "gbk"  # Default encoding for document files, None for auto-detection

# Logging configuration
LOG_LEVEL = "INFO"  # Default logging level
LOG_FORMAT = "%(asctime)s - %(funcName)s: %(lineno)d - %(levelname)s: %(message)s"  # Log message format
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB - Maximum size per log file before rotation
LOG_BACKUP_COUNT = 10  # Number of backup log files to keep
LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../logs"
)  # Log storage directory, None to disable file logging
LOG_TO_CONSOLE = True  # Whether to log to console

# Query settings
CACHE_SIZE = 100  # Cache size for boolean queries
QUERY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../test/test_query_0.txt"
)  # Query file path, None for input-based queries
PRINT_RESULTS = True  # Whether to print query results
RESULTS_PATH = None  # Path to save query results, None to disable saving

# Ollama model settings
OLLAMA_RAG_MAX_DOCS = 10  # Maximum number of documents to retrieve for each query

# System prompt for generating boolean queries from natural language questions
OLLAMA_SYS_PROMPT_GENERATE_BOOL_QUERY_FROM_QUESTION = """
You are a search query generator. Given a user question, generate a boolean search query that would help find relevant documents to answer the question.

Rules:
1. Use && for AND operations (terms that must appear together)
2. Use || for OR operations (alternative terms)
3. Use ! for NOT operations (exclude terms)
4. Use parentheses for grouping when needed
5. When query keywords (not logical operators) contain '&', '|', '!', '(', ')', use escape character ('\' + original character) to prevent confusion with logical operators
6. Focus on key concepts and terms from the question
7. Consider synonyms and related terms
8. Keep the query concise but comprehensive
9. Please give the query in a single line without any additional text
10. Do not include any explanations or comments, just the query
"""

# System prompt for generating answers based on retrieved documents
OLLAMA_SYS_PROMPT_GENERATE_ANSWER = """
You are an answer generator. Given a boolean query, generate a concise and informative answer based on the retrieved documents. The given documents are formatted like this json format:
```json
{
    "Question": "quest",
    "Documents": [
        "content1",
        "content2",
        "content3",
        ...
    ]
}
```
The retrieved documents may be empty (empty list for key "Documents"), in which case you should return an appropriate message indicating no relevant documents were found.
Some times there are no documents provided (no "Documents" key), you should return a message indicating that no documents were provided, answer the question based on your knowledge, or state that you cannot answer the question without documents.
"""

if __name__ == "__main__":
    # Example usage and testing of the configuration
    print("=== Document Collection Settings ===")
    print("Document Directory:", DOCS_DIR)
    print("Document Extensions:", DOC_EXTENSIONS)
    print("Document Extensions Exclude:", DOC_EXTENSIONS_EXCLUDE)
    print("Document Encoding:", DOC_ENCODING)

    print("\n=== Logging Configuration ===")
    print("Log Level:", LOG_LEVEL)
    print("Log Format:", LOG_FORMAT)
    print("Log Max Bytes:", LOG_MAX_BYTES)
    print("Log Backup Count:", LOG_BACKUP_COUNT)
    print("Log Directory:", LOG_DIR)
    print("Log to Console:", LOG_TO_CONSOLE)

    print("\n=== Query Settings ===")
    print("Cache Size for Queries:", CACHE_SIZE)
    print("Query File Path:", QUERY_FILE)
    print("Print Results:", PRINT_RESULTS)
    print("Results Save Path:", RESULTS_PATH)

    print("\n=== Ollama Configuration ===")
    print("Max Documents for RAG:", OLLAMA_RAG_MAX_DOCS)
    print(
        "Boolean Query Generation Prompt Length:",
        len(OLLAMA_SYS_PROMPT_GENERATE_BOOL_QUERY_FROM_QUESTION),
    )
    print("Answer Generation Prompt Length:", len(OLLAMA_SYS_PROMPT_GENERATE_ANSWER))
    print(
        "Boolean Query Prompt Preview (first 100 chars):",
        OLLAMA_SYS_PROMPT_GENERATE_BOOL_QUERY_FROM_QUESTION[:100] + "...",
    )
    print(
        "Answer Generation Prompt Preview (first 100 chars):",
        OLLAMA_SYS_PROMPT_GENERATE_ANSWER[:100] + "...",
    )

    # Validate configuration
    print("\n=== Configuration Validation ===")
    print("Documents directory exists:", os.path.exists(DOCS_DIR))
    print(
        "Query file exists:", os.path.exists(QUERY_FILE) if QUERY_FILE else "N/A (None)"
    )
    print("Log directory exists:", os.path.exists(LOG_DIR) if LOG_DIR else "N/A (None)")
