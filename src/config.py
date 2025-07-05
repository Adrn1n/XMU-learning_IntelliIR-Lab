"""
Global configuration settings for the Information Retrieval System.

This module contains all configuration parameters including:
- Document collection settings (directory, file extensions, encoding)
- Logging configuration (format, level, rotation, output destinations)
- Query processing settings (cache size, input source, output options)
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
LOG_FORMAT = "%(asctime)s - %(funcName)s: %(lineno)d - %(levelname)s: %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 10  # Number of backup log files
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

if __name__ == "__main__":
    # Example usage of the configuration
    print("Document Directory:", DOCS_DIR)
    print("Document Extensions:", DOC_EXTENSIONS)
    print("Log Level:", LOG_LEVEL)
    print("Log Format:", LOG_FORMAT)
    print("Cache Size for Queries:", CACHE_SIZE)
    print("Query File Path:", QUERY_FILE)
    print("Print Results:", PRINT_RESULTS)
    print("Results Save Path:", RESULTS_PATH)
    print("Log Directory:", LOG_DIR)
    print("Log Max Bytes:", LOG_MAX_BYTES)
    print("Log Backup Count:", LOG_BACKUP_COUNT)
    print("Log to Console:", LOG_TO_CONSOLE)
