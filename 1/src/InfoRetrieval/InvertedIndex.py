"""
Inverted Index implementation for information retrieval.

This module provides an inverted index data structure that maps terms to
the documents containing them. It supports document addition, removal,
and querying operations with efficient storage and retrieval.
"""

import sys
import os
from typing import Dict, List, Set, Union
from bidict import bidict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger
from utils.FileLoader import FileLoader

# from utils.Tokenizer import Tokenizer


class InvertedIndex:
    """
    Inverted index data structure for document retrieval.

    This class maintains a mapping from terms to documents containing them,
    enabling efficient full-text search operations. It uses a bidirectional
    mapping between document IDs and file paths for quick lookups.

    Attributes:
        __biject_docId_Path: Bidirectional mapping between document IDs and file paths
        __inverted_index: Dictionary mapping terms to [frequency, document_set] pairs
    """

    __logger = setup_logger(__name__)

    def __init__(
        self,
        docs_dir: str,
        docs_extensions: Union[List[str], None] = None,
        docs_extensions_exclude: Union[List[str], None] = None,
        docs_encoding: Union[str, None] = None,
    ):
        """
        Initialize the inverted index from a document directory.

        Args:
            docs_dir (str): Directory containing documents to index.
            docs_extensions (List[str], optional): File extensions to include.
            docs_extensions_exclude (List[str], optional): File extensions to exclude.
            docs_encoding (str, optional): Text encoding for documents.

        Raises:
            FileNotFoundError: If the document directory doesn't exist.
            PermissionError: If the directory cannot be accessed.
        """
        self.__biject_docId_Path: bidict[int, str] = bidict()
        self.__inverted_index: Dict[str, List[Union[int, Set[int]]]] = {}

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
            self.build_inv_idx(file_paths, docs_encoding)

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
        #     return [word for word in tokens if word.isalpha()]
        # except Exception as e:
        #     cls.__logger.error(f"Tokenization failed: {str(e)}")
        #     return []

        return [
            word for word in text.split() if word.isalpha()
        ]  # for teacher's testing

    def add_document(self, file_path: str, encoding: Union[str, None] = None) -> int:
        """
        Add a document to the inverted index.

        Args:
            file_path: Path to the document file
            encoding: Text encoding for the document

        Returns:
            Assigned document ID, or -1 if failed

        Raises:
            FileNotFoundError: If the document file does not exist
            IOError: If the file cannot be read
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

        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in self.__inverted_index:
                self.__inverted_index[token] = [0, set()]
            self.__inverted_index[token][0] += 1
            self.__inverted_index[token][1].add(doc_id)

        self.__logger.debug(
            f"Added document {file_path} with ID {doc_id}, {len(unique_tokens)} unique terms"
        )
        return doc_id

    def remove_document(self, identifier: Union[int, str]) -> Union[int, None]:
        """
        Remove a document from the inverted index.

        Args:
            identifier: Document ID or file path

        Returns:
            Document ID of removed document, or None if not found
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

        if doc_id is None:
            self.__logger.warning(f"Document not found: {identifier}")
            return None

        tokens_to_remove = []
        for token, data in self.__inverted_index.items():
            if doc_id in data[1]:
                data[1].remove(doc_id)
                data[0] -= 1
                if data[0] == 0:
                    tokens_to_remove.append(token)

        for token in tokens_to_remove:
            del self.__inverted_index[token]

        self.__logger.info(f"Removed document: {file_path}, ID: {doc_id}")
        return doc_id

    def get_docs_count(self) -> int:
        """
        Get the current number of documents in the index.

        Returns:
            Number of documents
        """
        return len(self.__biject_docId_Path)

    def get_keywords_count(self) -> int:
        """
        Get the current number of keywords in the index.

        Returns:
            Number of keywords
        """
        return len(self.__inverted_index)

    def query_document(self, identifier: Union[int, str]) -> Union[str, int, None]:
        """
        Query document by ID or file path.

        Args:
            identifier: Document ID or file path

        Returns:
            Corresponding file path or document ID, None if not found
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

    def build_inv_idx(
        self, file_paths: List[str], encoding: Union[str, None] = None
    ) -> Dict[str, List[Union[int, Set[int]]]]:
        """
        Build inverted index from document file paths.

        Args:
            file_paths: List of document file paths
            encoding: Text encoding for documents

        Returns:
            Inverted index dictionary
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

    def get_postings__const(
        self, token: str
    ) -> Union[List[Union[int, Set[int]]], None]:
        """
        Get posting list for specified token.

        Args:
            token: Keyword to search

        Returns:
            Tuple containing document frequency and document ID set, or None if not found

        Note: Returns reference to internal data for performance. Do not modify.
        """
        return self.__inverted_index.get(token, None)


if __name__ == "__main__":
    # Interactive test mode for inverted index
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        print("=== Inverted Index Interactive Test ===")
        invIndex = InvertedIndex(testDir, testExtensions, testExtensionsExclude)
        print(
            f"Index initialized with {invIndex.get_docs_count()} documents and {invIndex.get_keywords_count()} keywords"
        )

        while True:
            print("\n--- Menu ---")
            print("1. Show index statistics")
            print("2. Query document by ID")
            print("3. Query document by path")
            print("4. Add document")
            print("5. Remove document by ID")
            print("6. Remove document by path")
            print("7. Search keyword")
            print("8. List all documents")
            print("0. Exit")

            try:
                choice = input("\nEnter your choice (0-8): ").strip()

                if choice == "0":
                    print("Goodbye!")
                    break

                elif choice == "1":
                    print(f"Documents: {invIndex.get_docs_count()}")
                    print(f"Keywords: {invIndex.get_keywords_count()}")

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
                        freq, doc_set = postings
                        print(
                            f"Keyword '{keyword}' found in {len(doc_set)} documents (frequency: {freq})"
                        )
                        print(f"Document IDs: {sorted(list(doc_set))}")
                    else:
                        print(f"Keyword '{keyword}' not found")

                elif choice == "8":
                    if invIndex.get_docs_count() == 0:
                        print("No documents in index")
                    else:
                        print("All documents in index:")
                        for docId in range(1, invIndex.get_docs_count() + 1):
                            path = invIndex.query_document(docId)
                            if path:
                                print(f"  ID {docId}: {path}")

                else:
                    print("Invalid choice. Please enter a number between 0-8.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user.")
                break
            except Exception as excpt:
                print(f"Error: {str(excpt)}")

    except Exception as excpt:
        print(f"Failed to initialize inverted index: {str(excpt)}")
        import traceback

        traceback.print_exc()
