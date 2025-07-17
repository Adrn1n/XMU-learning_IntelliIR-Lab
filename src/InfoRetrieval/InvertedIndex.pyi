"""
Enhanced Inverted Index implementation for TF-IDF based information retrieval.

This module provides a sophisticated inverted index data structure that maps terms to documents with raw term frequency (raw_tf) tracking. The index is specifically designed to support TF-IDF calculations for advanced boolean retrieval systems and information retrieval applications.

Core Features:
- Raw term frequency (raw_tf) tracking for each term-document pair
- Document frequency (DF) statistics for IDF calculation
- Document total term count tracking for TF normalization
- Bidirectional document ID-path mapping for efficient lookups
- TF-IDF computation preparation and support
- Memory-efficient inverted index structure
- Comprehensive document management operations
- Document and term enumeration with get_all_documents__const() and get_all_terms()

Data Structures:
- Inverted index: term -> [document_frequency, {doc_id: raw_term_frequency}]
- Document mapping: bidirectional doc_id <-> file_path mapping
- Document term counts: doc_id -> total_term_count for TF normalization
"""

import logging
from typing import Dict, List, Optional, Union
from bidict import bidict

class InvertedIndex:
    """
    Enhanced inverted index for TF-IDF based document retrieval.

    This class maintains a sophisticated mapping from terms to documents with raw term frequency tracking, enabling efficient TF-IDF calculations and advanced boolean query processing. The index structure supports both document frequency (DF) and term frequency (TF) statistics, along with document-level term count tracking for normalized TF calculations.

    Data Structures:
        - __biject_docId_Path: Bidirectional mapping between numeric document IDs and file paths for efficient lookups
        - __inverted_index: Dictionary mapping each term to [DF, {doc_id: raw_tf}] where DF is document frequency and raw_tf is raw term frequency
        - __dict_docId_tCnt: Dictionary mapping document IDs to total term counts for TF normalization in TF-IDF calculations

    Attributes:
        __biject_docId_Path (bidict): Bidirectional document ID-path mapping
        __inverted_index (Dict): Term -> [DF, {doc_id: raw_tf}] mapping
        __dict_docId_tCnt (Dict): Document ID -> total term count mapping
    """

    __logger: logging.Logger
    __biject_docId_Path: bidict[int, str]
    __inverted_index: Dict[str, List[Union[int, Dict[int, int]]]]
    __dict_docId_tCnt: Dict[int, int]

    def __init__(
        self,
        docs_dir: str,
        docs_extensions: Optional[List[str]] = None,
        docs_extensions_exclude: Optional[List[str]] = None,
        docs_encoding: Optional[str] = None,
    ) -> None:
        """
        Initialize the enhanced inverted index from a document directory.

        Creates a TF-IDF ready inverted index by processing all documents in the specified directory. The index tracks both document frequency (DF) and raw term frequency (TF) for each term-document pair, providing essential data for TF-IDF calculations in boolean query processing.

        Args:
            docs_dir (str): Directory path containing documents to index
            docs_extensions (List[str], optional): File extensions to include. If None, all files are processed
            docs_extensions_exclude (List[str], optional): File extensions to exclude. Takes precedence over inclusion list
            docs_encoding (str, optional): Character encoding for document files. If None, auto-detection is used

        Raises:
            FileNotFoundError: If the document directory doesn't exist
            PermissionError: If the directory cannot be accessed
            ValueError: If directory path is invalid
        """
        ...

    @classmethod
    def __tokenize(cls, text: str) -> List[str]:
        """
        Tokenize text into individual words, filtering alphabetic tokens only.

        This method uses an external tokenizer for text processing with automatic language detection support, then filters the results to keep only alphabetic tokens (excluding numbers, punctuation, and symbols).

        Args:
            text (str): Input text to tokenize

        Returns:
            List[str]: List of alphabetic tokens extracted from the input text. Returns empty list if tokenization fails or no alphabetic tokens found.

        Note:
            - Supports multilingual text processing with automatic language detection
            - Logs errors if tokenization fails and returns empty list as fallback
        """
        ...

    def add_document(self, file_path: str, encoding: Optional[str] = None) -> int:
        """
        Add a document to the inverted index with raw term frequency tracking.

        This method processes a document and updates the inverted index with both document frequency (DF) and raw term frequency (raw_tf) information for each term, enabling TF-IDF calculations for boolean query processing.

        Args:
            file_path (str): Path to the document file to be indexed encoding (str, optional): Text encoding for the document file. If None, auto-detection will be attempted
            encoding (str, optional): Text encoding for the document file. If None, auto-detection will be attempted

        Returns:
            int: Assigned document ID, or -1 if indexing failed

        Raises:
            FileNotFoundError: If the document file does not exist
            IOError: If the file cannot be read due to permission or encoding issues

        Note:
            - Updates both DF and raw_tf for each term
            - Automatically assigns sequential document IDs
            - Skips duplicate documents and returns existing document ID
        """
        ...

    def remove_document(self, identifier: Union[int, str]) -> Optional[int]:
        """
        Remove a document from the inverted index and update term statistics.

        This method removes a document and its associated term frequencies from the inverted index, properly updating document frequency (DF) counts for all affected terms. Terms with zero document frequency are automatically removed.

        Args:
            identifier (Union[int, str]): Document ID or file path to remove

        Returns:
            Union[int, None]: Document ID of removed document, or None if not found

        Note:
            - Automatically decrements document frequency for all terms in the removed document
            - Removes terms with zero document frequency from the index
            - Maintains index consistency for accurate TF-IDF calculations
        """
        ...

    def __build_inv_idx(
        self, file_paths: List[str], encoding: Optional[str] = None
    ) -> Dict[str, List[Union[int, Dict[int, int]]]]:
        """
        Build inverted index from document file paths with raw term frequency tracking.

        This method processes multiple documents and builds a complete inverted index with both document frequency (DF) and raw term frequency (raw_tf) information for each term-document pair, preparing the index for TF-IDF calculations.

        Args:
            file_paths (List[str]): List of document file paths to process
            encoding (str, optional): Text encoding for documents

        Returns:
            Dict[str, List[Union[int, Dict[int, int]]]]: The constructed inverted index
        """
        ...

    def get_all_documents__const(self) -> bidict[int, str]:
        """
        Get the bidirectional mapping of all documents indexed in the inverted index.

        Returns:
            bidict[int, str]: Bidirectional mapping between document IDs and file paths

        Note:
            Returns reference to internal data for performance. Do not modify the returned data.
        """
        ...

    def get_all_terms(self) -> List[str]:
        """
        Get a list of all terms in the inverted index.

        Returns:
            List[str]: List of all unique terms indexed

        Note:
            Returns a new list copy of the terms for safety. Safe to modify the returned data.
        """
        ...

    def query_document(self, identifier: Union[int, str]) -> Optional[Union[str, int]]:
        """
        Query document by ID or file path.

        Args:
            identifier (Union[int, str]): Document ID or file path

        Returns:
            Union[str, int, None]: Corresponding file path or document ID, None if not found
        """
        ...

    def get_docs_total_term_count(self, identifier: Union[int, str]) -> int:
        """
        Get the total term count for a document by ID or file path.

        This method retrieves the total number of terms (tokens) in a specific document, which is essential for TF-IDF calculations, particularly for normalized term frequency computations. The total term count includes all terms processed during document indexing, including repeated occurrences.

        Args:
            identifier (Union[int, str]): Document ID (int) or file path (str)

        Returns:
            int: Total term count for the specified document, or -1 if document not found

        Note:
            - Total term count is used for TF normalization: normalized_tf = raw_tf / total_terms
            - This is crucial for computing accurate TF-IDF scores in information retrieval
            - The count includes all alphabetic tokens processed during document indexing
        """
        ...

    def get_postings__const(
        self, token: str
    ) -> Optional[List[Union[int, Dict[int, int]]]]:
        """
        Get posting list for specified token with raw term frequency data.

        This method returns the complete posting information for a token, including document frequency (DF) and raw term frequency (raw_tf) for each document, which is essential for TF-IDF calculations in boolean queries.

        Args:
            token (str): Keyword to search

        Returns:
            Union[List[Union[int, Dict[int, int]]], None]: List containing [DF, {doc_id: raw_tf}] if token exists, None otherwise
                - DF: Document frequency (number of documents containing the token)
                - {doc_id: raw_tf}: Dictionary mapping document IDs to raw term frequencies

        Note:
            Returns reference to internal data for performance. Do not modify the returned data.
        """
        ...
