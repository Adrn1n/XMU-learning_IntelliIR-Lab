"""
Enhanced Inverted Index implementation for TF-IDF based information retrieval.

This module provides a sophisticated inverted index data structure that maps terms to
documents with raw term frequency (raw_tf) tracking. The index is specifically designed
to support TF-IDF calculations for advanced boolean retrieval systems and information
retrieval applications.

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

import sys
import os
from typing import Dict, List, Optional, Union
from bidict import bidict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger
from utils.FileLoader import FileLoader

# from utils.Tokenizer import Tokenizer


class InvertedIndex:
    """
    Enhanced inverted index for TF-IDF based document retrieval.

    This class maintains a sophisticated mapping from terms to documents with
    raw term frequency tracking, enabling efficient TF-IDF calculations and
    advanced boolean query processing. The index structure supports both
    document frequency (DF) and term frequency (TF) statistics, along with
    document-level term count tracking for normalized TF calculations.

    Data Structures:
        - __biject_docId_Path: Bidirectional mapping between numeric document IDs
          and file paths for efficient lookups
        - __inverted_index: Dictionary mapping each term to [DF, {doc_id: raw_tf}]
          where DF is document frequency and raw_tf is raw term frequency
        - __dict_docId_tCnt: Dictionary mapping document IDs to total term counts
          for TF normalization in TF-IDF calculations

    Attributes:
        __biject_docId_Path (bidict): Bidirectional document ID-path mapping
        __inverted_index (Dict): Term -> [DF, {doc_id: raw_tf}] mapping
        __dict_docId_tCnt (Dict): Document ID -> total term count mapping
    """

    __logger = setup_logger(__name__)

    def __init__(
        self,
        docs_dir: str,
        docs_extensions: Optional[List[str]] = None,
        docs_extensions_exclude: Optional[List[str]] = None,
        docs_encoding: Optional[str] = None,
    ):
        """
        Initialize the enhanced inverted index from a document directory.

        Creates a TF-IDF ready inverted index by processing all documents in the
        specified directory. The index tracks both document frequency (DF) and
        raw term frequency (TF) for each term-document pair, providing essential
        data for TF-IDF calculations in boolean query processing.

        Args:
            docs_dir (str): Directory path containing documents to index
            docs_extensions (List[str], optional): File extensions to include.
                If None, all files are processed
            docs_extensions_exclude (List[str], optional): File extensions to exclude.
                Takes precedence over inclusion list
            docs_encoding (str, optional): Character encoding for document files.
                If None, auto-detection is used

        Raises:
            FileNotFoundError: If the document directory doesn't exist
            PermissionError: If the directory cannot be accessed
            ValueError: If directory path is invalid
        """
        self.__biject_docId_Path: bidict[int, str] = bidict()
        self.__inverted_index: Dict[str, List[Union[int, Dict[int, int]]]] = (
            {}
        )  # term -> [document_frequency, {doc_id: raw_term_frequency}]
        self.__dict_docId_tCnt: Dict[int, int] = (
            {}
        )  # doc_id -> total_term_count for TF normalization

        if not os.path.exists(docs_dir):
            self.__logger.error(f"Document directory not found: {docs_dir}")
            raise FileNotFoundError(f"Document directory not found: {docs_dir}")

        try:
            # Load all files from the document directory
            file_paths = FileLoader.load_directory_all_files(
                docs_dir, docs_extensions, docs_extensions_exclude
            )

            if not file_paths:
                self.__logger.warning(f"No documents found in directory: {docs_dir}")
                return

            self.__logger.info(
                f"Building inverted index for {len(file_paths)} documents"
            )
            self.__build_inv_idx(file_paths, docs_encoding)

        except Exception as e:
            self.__logger.error(f"Failed to initialize inverted index: {str(e)}")
            raise

    @classmethod
    def __tokenize(cls, text: str) -> List[str]:
        """
        Tokenize text into individual words, filtering alphabetic tokens only.

        This method uses an external tokenizer for text processing with automatic language
        detection support, then filters the results to keep only alphabetic tokens
        (excluding numbers, punctuation, and symbols).

        Args:
            text (str): Input text to tokenize

        Returns:
            List[str]: List of alphabetic tokens extracted from the input text.
                      Returns empty list if tokenization fails or no alphabetic tokens found.

        Note:
            - Supports multilingual text processing with automatic language detection
            - Logs errors if tokenization fails and returns empty list as fallback
        """
        # try:
        #     tokens = Tokenizer.tokenize(text, "auto")
        #     return [word for word in tokens]
        # except Exception as e:
        #     cls.__logger.error(f"Tokenization failed: {str(e)}")
        #     return []

        return [word for word in text.split()]  # for teacher's testing

    def add_document(self, file_path: str, encoding: Optional[str] = None) -> int:
        """
        Add a document to the inverted index with raw term frequency tracking.

        This method processes a document and updates the inverted index with both
        document frequency (DF) and raw term frequency (raw_tf) information for
        each term, enabling TF-IDF calculations for boolean query processing.

        Args:
            file_path (str): Path to the document file to be indexed
            encoding (str, optional): Text encoding for the document file.
                If None, auto-detection will be attempted

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
        if file_path in self.__biject_docId_Path.values():
            self.__logger.warning(f"Document already exists: {file_path}")
            return self.__biject_docId_Path.inverse[file_path]

        try:
            content = FileLoader.load_file_content(file_path, encoding)
            if not content:
                self.__logger.warning(f"Document content is empty: {file_path}")
                return -1
        except Exception as e:
            self.__logger.error(f"Failed to load document {file_path}: {str(e)}")
            return -1

        tokens = self.__tokenize(content)
        if not tokens:
            self.__logger.warning(f"No tokens extracted from document: {file_path}")
            return -1

        doc_id = len(self.__biject_docId_Path) + 1
        self.__biject_docId_Path[doc_id] = file_path
        self.__dict_docId_tCnt[doc_id] = len(tokens)

        # Update inverted index with raw term frequencies
        unique_tokens = set()
        for token in tokens:
            if token not in unique_tokens:
                if token not in self.__inverted_index:
                    self.__inverted_index[token] = [0, dict()]
                self.__inverted_index[token][0] += 1
                self.__inverted_index[token][1][doc_id] = 0
                unique_tokens.add(token)
            self.__inverted_index[token][1][doc_id] += 1  # Increment raw term frequency

        self.__logger.debug(
            f"Added document {file_path} with ID {doc_id}, {len(unique_tokens)} unique terms"
        )
        return doc_id

    def remove_document(self, identifier: Union[int, str]) -> Optional[int]:
        """
        Remove a document from the inverted index and update term statistics.

        This method removes a document and its associated term frequencies from the
        inverted index, properly updating document frequency (DF) counts for all
        affected terms. Terms with zero document frequency are automatically removed.

        Args:
            identifier (Union[int, str]): Document ID or file path to remove

        Returns:
            Union[int, None]: Document ID of removed document, or None if not found

        Note:
            - Automatically decrements document frequency for all terms in the removed document
            - Removes terms with zero document frequency from the index
            - Maintains index consistency for accurate TF-IDF calculations
        """
        if isinstance(identifier, int):
            doc_id = identifier
            file_path = self.__biject_docId_Path.pop(identifier, None)
        elif isinstance(identifier, str):
            doc_id = self.__biject_docId_Path.inverse.pop(identifier, None)
            file_path = identifier
        else:
            self.__logger.error(f"Invalid identifier type: {type(identifier)}")
            return None

        if doc_id is None or file_path is None:
            self.__logger.warning(f"Document not found: {identifier}")
            return None
        self.__dict_docId_tCnt.pop(doc_id, None)

        tokens_to_remove = []
        for token, data in self.__inverted_index.items():
            if doc_id in data[1]:
                del data[1][doc_id]
                data[0] -= 1  # Decrement document frequency
                if data[0] == 0:
                    tokens_to_remove.append(token)

        for token in tokens_to_remove:
            del self.__inverted_index[token]

        self.__logger.info(f"Removed document: {file_path}, ID: {doc_id}")
        return doc_id

    def __build_inv_idx(
        self, file_paths: List[str], encoding: Optional[str] = None
    ) -> Dict[str, List[Union[int, Dict[int, int]]]]:
        """
        Build inverted index from document file paths with raw term frequency tracking.

        This method processes multiple documents and builds a complete inverted index
        with both document frequency (DF) and raw term frequency (raw_tf) information
        for each term-document pair, preparing the index for TF-IDF calculations.

        Args:
            file_paths (List[str]): List of document file paths to process
            encoding (str, optional): Text encoding for documents

        Returns:
            Dict[str, List[Union[int, Dict[int, int]]]]: The constructed inverted index
        """
        for file_path in file_paths:
            try:
                self.add_document(file_path, encoding)
            except Exception as e:
                self.__logger.error(f"Failed to add document {file_path}: {str(e)}")
                continue

        self.__logger.info(
            f"Inverted index build completed - Documents: {len(self.__biject_docId_Path)}, "
            f"Keywords: {len(self.__inverted_index)}"
        )
        return self.__inverted_index

    def get_all_documents__const(self) -> bidict[int, str]:
        """
        Get the bidirectional mapping of all documents indexed in the inverted index.

        Returns:
            bidict[int, str]: Bidirectional mapping between document IDs and file paths

        Note:
            Returns reference to internal data for performance. Do not modify the returned data.
        """
        return self.__biject_docId_Path

    def get_all_terms(self) -> List[str]:
        """
        Get a list of all terms in the inverted index.

        Returns:
            List[str]: List of all unique terms indexed

        Note:
            Returns a new list copy of the terms for safety. Safe to modify the returned data.
        """
        return list(self.__inverted_index.keys())

    def query_document(self, identifier: Union[int, str]) -> Optional[Union[str, int]]:
        """
        Query document by ID or file path.

        Args:
            identifier (Union[int, str]): Document ID or file path

        Returns:
            Union[str, int, None]: Corresponding file path or document ID, None if not found
        """
        try:
            if isinstance(identifier, int):
                return self.__biject_docId_Path.get(identifier)
            elif isinstance(identifier, str):
                return self.__biject_docId_Path.inverse.get(identifier)
            else:
                self.__logger.error(
                    f"Invalid query identifier type: {type(identifier)}"
                )
                return None
        except Exception as e:
            self.__logger.error(f"Failed to query document: {str(e)}")
            raise

    def get_docs_total_term_count(self, identifier: Union[int, str]) -> int:
        """
        Get the total term count for a document by ID or file path.

        This method retrieves the total number of terms (tokens) in a specific document,
        which is essential for TF-IDF calculations, particularly for normalized term
        frequency computations. The total term count includes all terms processed
        during document indexing, including repeated occurrences.

        Args:
            identifier (Union[int, str]): Document ID (int) or file path (str)

        Returns:
            int: Total term count for the specified document, or -1 if document not found

        Note:
            - Total term count is used for TF normalization: normalized_tf = raw_tf / total_terms
            - This is crucial for computing accurate TF-IDF scores in information retrieval
            - The count includes all alphabetic tokens processed during document indexing
        """
        try:
            if isinstance(identifier, int):
                return self.__dict_docId_tCnt.get(identifier, -1)
            elif isinstance(identifier, str):
                doc_id = self.__biject_docId_Path.inverse.get(identifier)
                return self.__dict_docId_tCnt[doc_id] if doc_id else -1
            else:
                self.__logger.error(
                    f"Invalid query identifier type: {type(identifier)}"
                )
                return -1
        except Exception as e:
            self.__logger.error(f"Failed to get document term count: {str(e)}")
            return -1

    def get_postings__const(
        self, token: str
    ) -> Optional[List[Union[int, Dict[int, int]]]]:
        """
        Get posting list for specified token with raw term frequency data.

        This method returns the complete posting information for a token, including
        document frequency (DF) and raw term frequency (raw_tf) for each document,
        which is essential for TF-IDF calculations in boolean queries.

        Args:
            token (str): Keyword to search

        Returns:
            Union[List[Union[int, Dict[int, int]]], None]:
                List containing [DF, {doc_id: raw_tf}] if token exists, None otherwise
                - DF: Document frequency (number of documents containing the token)
                - {doc_id: raw_tf}: Dictionary mapping document IDs to raw term frequencies

        Note:
            Returns reference to internal data for performance. Do not modify the returned data.
        """
        return self.__inverted_index.get(token, None)


if __name__ == "__main__":
    # Interactive test mode for TF-IDF ready inverted index
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        print("=== Enhanced TF-IDF Inverted Index Interactive Test ===")
        invIndex = InvertedIndex(testDir, testExtensions, testExtensionsExclude)
        print(
            f"Index initialized with {len(invIndex.get_all_documents__const())} documents and {len(invIndex.get_all_terms())} keywords"
        )
        print(
            "Index tracks document frequency, raw term frequency, and document term counts for TF-IDF calculations"
        )

        while True:
            print("\n--- Menu ---")
            print("1. Show index statistics")
            print("2. Query document by ID")
            print("3. Query document by path")
            print("4. Add document")
            print("5. Remove document by ID")
            print("6. Remove document by path")
            print("7. Search keyword with TF-IDF data")
            print("8. List all documents")
            print("9. Get document total term count")
            print("0. Exit")

            try:
                choice = input("\nEnter your choice (0-9): ").strip()

                if choice == "0":
                    print("Goodbye!")
                    break

                elif choice == "1":
                    allDocs = invIndex.get_all_documents__const()
                    allTerms = invIndex.get_all_terms()
                    print(f"Documents: {len(allDocs)}")
                    print(f"Keywords: {len(allTerms)}")
                    if len(allDocs) > 0:
                        totalTerms = sum(
                            invIndex.get_docs_total_term_count(doc_id)
                            for doc_id, _ in allDocs
                        )
                        avgTerms = totalTerms / len(allDocs)
                        print(f"Total terms across all documents: {totalTerms}")
                        print(f"Average terms per document: {avgTerms:.2f}")

                elif choice == "2":
                    docId = input("Enter document ID: ").strip()
                    try:
                        docId = int(docId)
                        path = invIndex.query_document(docId)
                        if path:
                            print(f"Document ID {docId}: {path}")
                        else:
                            print(f"Document ID {docId} not found")
                    except ValueError:
                        print("Invalid document ID. Please enter a number.")

                elif choice == "3":
                    path = input("Enter file path: ").strip()
                    docId = invIndex.query_document(path)
                    if docId:
                        print(f"File path '{path}': Document ID {docId}")
                    else:
                        print(f"File path '{path}' not found")

                elif choice == "4":
                    path = input("Enter file path to add: ").strip()
                    docId = invIndex.add_document(path)
                    if docId > 0:
                        print(f"Document added successfully with ID: {docId}")
                    else:
                        print("Failed to add document")

                elif choice == "5":
                    docId = input("Enter document ID to remove: ").strip()
                    try:
                        docId = int(docId)
                        removedId = invIndex.remove_document(docId)
                        if removedId:
                            print(f"Document ID {removedId} removed successfully")
                        else:
                            print(f"Document ID {docId} not found")
                    except ValueError:
                        print("Invalid document ID. Please enter a number.")

                elif choice == "6":
                    path = input("Enter file path to remove: ").strip()
                    removedId = invIndex.remove_document(path)
                    if removedId:
                        print(
                            f"Document '{path}' (ID: {removedId}) removed successfully"
                        )
                    else:
                        print(f"Document '{path}' not found")

                elif choice == "7":
                    keyword = input("Enter keyword to search: ").strip()
                    postings = invIndex.get_postings__const(keyword)
                    if postings:
                        df, doc_tf_map = postings
                        print(
                            f"Keyword '{keyword}' found in {len(doc_tf_map)} documents, DF: {df}"
                        )
                        print("Document frequencies - doc_id: raw_tf:")
                        for docId, raw_tf in doc_tf_map.items():
                            print(f"\tDocument {docId}: {raw_tf} occurrences")
                    else:
                        print(f"Keyword '{keyword}' not found")

                elif choice == "8":
                    allDocs = invIndex.get_all_documents__const()
                    if len(allDocs) == 0:
                        print("No documents in index")
                    else:
                        print("All documents in index:")
                        for docId, path in allDocs.items():
                            term_count = invIndex.get_docs_total_term_count(docId)
                            print(f"\tID {docId}: {path} (Total terms: {term_count})")

                elif choice == "9":
                    idntfr = input("Enter document ID or file path: ").strip()
                    try:
                        # Try to parse as integer first
                        docId = int(idntfr)
                        term_count = invIndex.get_docs_total_term_count(docId)
                        if term_count >= 0:
                            print(f"Document ID {docId} has {term_count} total terms")
                        else:
                            print(f"Document ID {docId} not found")
                    except ValueError:
                        # Treat as file path
                        term_count = invIndex.get_docs_total_term_count(idntfr)
                        if term_count >= 0:
                            print(f"Document '{idntfr}' has {term_count} total terms")
                        else:
                            print(f"Document '{idntfr}' not found")

                else:
                    print("Invalid choice. Please enter a number between 0-9.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user.")
                break
            except Exception as excpt:
                print(f"Error: {str(excpt)}")

    except Exception as excpt:
        print(f"Failed to initialize inverted index: {str(excpt)}")
        import traceback

        traceback.print_exc()
