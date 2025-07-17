"""
Boolean retrieval implementation with vector space model scoring for information retrieval.

This module provides advanced boolean query processing capabilities including:
- Automatic tokenization of search content between logical operators
- Support for AND (&&), OR (||), and NOT (!) operators with vector space model scoring
- Parentheses for grouping complex expressions
- Multi-term query processing with implicit AND logic between tokenized terms
- Vector space model scoring using cosine similarity for document ranking
- Document result ranking based on vector space relevance scores
- LFU-LRU cache strategy for query optimization
- Expression parsing and evaluation with vector space model scoring support
- Sorted result output by cosine similarity relevance scores

The system processes queries by:
1. Parsing boolean expressions and extracting search content between logical operators
2. Tokenizing each search content segment into individual terms
3. Applying implicit AND logic to tokenized terms for multi-term queries
4. Computing document scores using vector space model with cosine similarity
5. Combining results using boolean operators while preserving relevance scores
"""

import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger

from InfoRetrieval.InvertedIndex import InvertedIndex

from utils.Tokenizer import Tokenizer
from InfoRetrieval.RankingWeight import RankingWeightCalculator


class BoolRetrieval:
    """
    Advanced Boolean retrieval system with vector space model scoring.

    This class implements a sophisticated boolean query processor that combines
    traditional boolean logic (AND, OR, NOT) with vector space model scoring using
    cosine similarity to provide ranked document retrieval results. The system
    automatically tokenizes search terms between logical operators and computes
    document relevance scores using vector space model.

    Key Features:
    - Automatic tokenization of search content between logical operators
    - Boolean operators with vector space scoring: AND (&&), OR (||), NOT (!)
    - Parentheses support for complex query expressions
    - Multi-term query processing with implicit AND logic between tokenized terms
    - Vector space model with TF-IDF weighting and cosine similarity scoring
    - LFU-LRU cache strategy for query performance optimization
    - Sorted result output by cosine similarity scores
    - Expression parsing with precedence handling

    Query Processing Workflow:
    1. Parse boolean expressions and identify search terms between operators
    2. Tokenize search terms for each logical component
    3. Apply implicit AND logic to multi-term tokenized segments
    4. Execute boolean logic to find candidate documents
    5. Compute vector space model scores using cosine similarity
    6. Return ranked results sorted by relevance scores

    Operator Semantics with Vector Space Model:
    - AND (&&): Returns intersection with minimum cosine similarity scores
    - OR (||): Returns union with maximum cosine similarity scores
    - NOT (!): Returns complement with computed vector space scores

    Cache Strategy:
    - Least Frequently Used (LFU) with Least Recently Used (LRU) tie-breaking
    - Configurable cache size for optimal memory usage
    - Hit count tracking and timestamp management
    """

    __logger = setup_logger(__name__)

    def __init__(
        self, identifier: Union[InvertedIndex, Dict[str, Any]], cache_size: int = 100
    ):
        """
        Initialize Boolean retrieval system with vector space model scoring support.

        Args:
            identifier: InvertedIndex instance or dict containing document directory path
                       and other configuration parameters for building the index
            cache_size: Cache size for query results with vector space scores, default 100

        Raises:
            ValueError: When identifier dict lacks required keys or invalid cache size
            TypeError: When identifier type is incorrect
            Exception: When other initialization errors occur
        """
        try:
            self.__logger.info(
                f"Initializing BoolRetrieval with cache size: {cache_size}"
            )

            if cache_size <= 0:
                raise ValueError(f"Cache size must be positive, got: {cache_size}")

            self.__cache_size = cache_size
            self.__cache: Dict[str, List[Union[int, List[Tuple[int, float]]]]] = (
                {}
            )  # query_str -> [timestamp, hit_count, [(doc_id, cosine_similarity_score), ...]]
            self.__inverted_index: InvertedIndex

            if isinstance(identifier, InvertedIndex):
                self.__inverted_index = identifier
                self.__logger.debug("Using existing InvertedIndex instance")
            elif isinstance(identifier, dict):
                if "docs_dir" in identifier:
                    self.__logger.debug(
                        f"Creating InvertedIndex from directory: {identifier.get('docs_dir')}"
                    )
                    self.__inverted_index = InvertedIndex(**identifier)
                else:
                    self.__logger.error("identifier dict missing 'docs_dir' key")
                    raise ValueError(
                        "identifier dict must contain 'docs_dir' key for InvertedIndex initialization"
                    )
            else:
                self.__logger.error(f"Unsupported identifier type: {type(identifier)}")
                raise TypeError(
                    "identifier must be InvertedIndex instance or dict with docs_dir key"
                )

            self.__logger.info(
                f"BoolRetrieval initialized successfully - Cache size: {self.__cache_size}, "
            )

        except Exception as e:
            self.__logger.error(f"BoolRetrieval initialization failed: {str(e)}")
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
        try:
            tokens = Tokenizer.tokenize(text, "auto")
            return [word for word in tokens]
        except Exception as e:
            cls.__logger.error(f"Tokenization failed: {str(e)}")
            return []

        # return [word for word in text.split()]  # for teacher's testing

    def __apply_operator(
        self,
        operand_stack: List[Dict[int, float]],
        operator_stack: List[str],
    ):
        """
        Apply top operator from operator stack with vector space model score computation.

        Implements boolean operations while preserving vector space model relevance scores:
        - AND (&&): Intersection with minimum cosine similarity scores (conservative scoring)
        - OR (||): Union with maximum cosine similarity scores (optimistic scoring)
        - NOT (!): Complement with computed vector space scores for non-matching documents

        Args:
            operand_stack: Stack of operands as {doc_id: cosine_similarity_score} dictionaries
            operator_stack: Stack of operators (&&, ||, !)

        Raises:
            IndexError: When stack is empty or insufficient operands
            ValueError: When unknown operator is encountered
        """
        try:
            if not operator_stack:
                self.__logger.warning("Operator stack is empty")
                return

            op = operator_stack.pop()
            self.__logger.debug(f"Applying operator: {op}")

            if op == "!":
                if not operand_stack:
                    self.__logger.error("NOT operation requires one operand")
                    raise IndexError("NOT operation requires one operand")

                operand = operand_stack.pop()
                all_docs = self.__inverted_index.get_all_documents__const().keys()
                all_docs_cnt = len(all_docs)
                res_len = all_docs_cnt - len(operand)
                if all_docs is None:
                    self.__logger.error("Failed to retrieve all documents from index")
                    raise ValueError("Failed to retrieve all documents from index")
                results = {}
                for doc_id in all_docs:
                    if doc_id not in operand:
                        results[doc_id] = RankingWeightCalculator.tf_idf(
                            self.__inverted_index.get_docs_total_term_count(doc_id),
                            res_len,
                            all_docs_cnt,
                        )
                operand_stack.append(results)
                self.__logger.debug(f"NOT operation result: {len(results)} documents")

            elif op == "&&":
                if len(operand_stack) < 2:
                    self.__logger.error(
                        f"AND operation requires two operands, got: {len(operand_stack)}"
                    )
                    raise IndexError("AND operation requires two operands")

                right = operand_stack.pop()
                left = operand_stack.pop()

                keys = left.keys() & right.keys()
                results = {}
                for k in keys:
                    results[k] = min(left[k], right[k])
                operand_stack.append(results)
                self.__logger.debug(f"AND operation result: {len(results)} documents")

            elif op == "||":
                if len(operand_stack) < 2:
                    self.__logger.error(
                        f"OR operation requires two operands, got: {len(operand_stack)}"
                    )
                    raise IndexError("OR operation requires two operands")

                right = operand_stack.pop()
                left = operand_stack.pop()

                keys = left.keys() | right.keys()
                results = {}
                for k in keys:
                    results[k] = max(left.get(k, 0), right.get(k, 0))
                operand_stack.append(results)
                self.__logger.debug(f"OR operation result: {len(results)} documents")

            else:
                self.__logger.error(f"Unknown operator: {op}")
                raise ValueError(f"Unsupported operator: {op}")

        except Exception as e:
            self.__logger.error(f"Failed to apply operator: {str(e)}")
            raise

    def __parse_keyword(self, keyword: str) -> Dict[int, float]:
        """
        Parse keyword and compute vector space model scored results.

        Processes a single keyword and returns documents with their corresponding
        vector space model relevance scores. The parsing workflow:
        1. Tokenizes the keyword into individual terms
        2. Applies implicit AND logic to tokenized terms for multi-term queries
        3. Computes document scores using vector space model with cosine similarity

        Args:
            keyword: Keyword string to search in the inverted index

        Returns:
            Dictionary mapping document IDs to their cosine similarity relevance scores
        """
        keyword = re.sub(r"\\([&|!()])", r"\1", keyword)
        self.__logger.debug(f"Processing keyword: {keyword}")

        # Step 1: Tokenize search content between logical operators
        words = self.__tokenize(keyword)

        # Step 2: Initialize vector space model computation structures
        keyword_results = {}
        weights_matrix = []  # Document-term weight matrix for vector space model
        query_weight_dict = (
            {}
        )  # Query term weights: {term: [tf_in_query, df_in_collection]}
        query_words = []
        doc_ids = []  # Document IDs that contain at least one query term

        all_docs_cnt = len(self.__inverted_index.get_all_documents__const())

        # Step 3: Build TF-IDF weighted vectors for query and documents
        for word in words:
            if not word:
                continue

            if word in query_weight_dict:
                # Increment term frequency in query
                query_words.append(word)
                query_weight_dict[word][0] += 1
            else:
                # Get postings for new term and build document vectors
                postings = self.__inverted_index.get_postings__const(word)
                if postings:
                    df = postings[0]
                    query_weight_dict[word] = [1, df]
                    weights_matrix.append([])
                    if doc_ids:
                        # Filter documents that contain current term (AND logic for multi-term queries)
                        tmp = []
                        for doc_id in doc_ids:
                            if doc_id in postings[1]:
                                tf = postings[1][doc_id]
                                tmp.append(doc_id)
                                # Calculate TF-IDF weight for document-term pair
                                weights_matrix[-1].append(
                                    RankingWeightCalculator.tf_idf(
                                        tf, df + 1, all_docs_cnt + 1
                                    )
                                )
                            else:
                                # Remove documents that don't contain current term
                                doc_idx = len(tmp)
                                for row in weights_matrix[:-1]:
                                    del row[doc_idx]
                        if not weights_matrix[0]:
                            # No documents contain all terms
                            query_weight_dict = {}
                            break
                        else:
                            doc_ids = tmp
                    else:
                        # First term: initialize document list
                        for doc_id, tf in postings[1].items():
                            doc_ids.append(doc_id)
                            weights_matrix[-1].append(
                                RankingWeightCalculator.tf_idf(
                                    tf, df + 1, all_docs_cnt + 1
                                )
                            )
                else:
                    self.__logger.debug(f"Keyword '{word}' not found in any documents")
                    query_weight_dict = {}
                    break

        # Step 4: Compute cosine similarity scores using vector space model
        if query_weight_dict:
            if len(query_weight_dict) == 1:
                # Single term query: use its weight directly
                for doc_id in doc_ids:
                    tf, df = query_weight_dict[next(iter(query_weight_dict))]
                    rating = RankingWeightCalculator.tf_idf(
                        tf, df + 1, all_docs_cnt + 1
                    )
                    keyword_results[doc_id] = rating
            else:
                # Build query vector with TF-IDF weights
                query_vector = [
                    RankingWeightCalculator.tf_idf(
                        query_weight_dict[word][0],
                        query_weight_dict[word][1] + 1,
                        all_docs_cnt + 1,
                    )
                    for word in query_weight_dict
                ]
                # Calculate cosine similarity for each candidate document
                for doc_id in doc_ids:
                    doc_vector = [
                        weights_matrix[doc_idx][doc_ids.index(doc_id)]
                        for doc_idx in range(len(weights_matrix))
                    ]
                    # Compute cosine similarity between query and document vectors
                    rating = RankingWeightCalculator.cosine_similarity(
                        query_vector, doc_vector
                    )
                    keyword_results[doc_id] = rating

        # Step 5: Log results and return keyword results
        if keyword_results:
            self.__logger.debug(
                f"Keyword '{keyword}' processed, results: {len(keyword_results)} documents"
            )
        else:
            self.__logger.debug(f"Keyword '{keyword}' processed, no results found")

        return keyword_results

    def __parse_expression(self, expr: str) -> Dict[int, float]:
        """
        Parse boolean expression and compute vector space model scored results using stacks.

        Processes boolean expressions with operator precedence and returns documents
        with their corresponding vector space model relevance scores. The parsing workflow:
        1. Parses boolean expressions and extracts search content between logical operators
        2. Tokenizes each search content segment into individual terms
        3. Applies implicit AND logic to tokenized terms for multi-term queries
        4. Computes document scores using vector space model with cosine similarity
        5. Handles operator precedence: ! (NOT) > && (AND) > || (OR)
        6. Supports parentheses for expression grouping and special character escaping

        Args:
            expr: Boolean expression string with keywords and operators

        Returns:
            Dictionary mapping document IDs to their cosine similarity relevance scores

        Raises:
            ValueError: When expression format is invalid or contains syntax errors
            Exception: When parsing errors occur during evaluation
        """
        try:
            if not expr or not expr.strip():
                self.__logger.warning("Query expression is empty")
                return dict()

            self.__logger.info(f"Parsing expression: {expr}")

            tokens = re.findall(r"(?:[^\\&|!()]|\\.)+|&&|\|\||!|\(|\)", expr)
            self.__logger.debug(f"Tokenization result: {tokens}")

            if not tokens:
                self.__logger.warning("Expression tokenization resulted in empty list")
                return dict()

            operand_stack = []
            operator_stack = []
            precedence = {"(": 0, "||": 1, "&&": 2, "!": 3}

            for token in tokens:
                token = token.strip()
                try:
                    self.__logger.debug(f"Processing token: {token}")

                    if token == "(":
                        operator_stack.append(token)
                    elif token == ")":
                        bracket_processed = False
                        while operator_stack and operator_stack[-1] != "(":
                            self.__apply_operator(operand_stack, operator_stack)
                        if operator_stack:
                            operator_stack.pop()
                            bracket_processed = True
                        if not bracket_processed:
                            self.__logger.error(
                                "Right parenthesis without matching left parenthesis"
                            )
                            raise ValueError(
                                "Mismatched parentheses: missing left parenthesis"
                            )
                    elif token in ["!", "&&", "||"]:
                        while (
                            operator_stack
                            and operator_stack[-1] != "("
                            and precedence.get(operator_stack[-1], 0)
                            >= precedence[token]
                        ):
                            self.__apply_operator(operand_stack, operator_stack)
                        operator_stack.append(token)
                    else:
                        if not token:
                            continue

                        operand_stack.append(self.__parse_keyword(token))
                except Exception as e:
                    self.__logger.error(f"Error processing token '{token}': {str(e)}")
                    raise

            if "(" in operator_stack:
                self.__logger.error("Unmatched left parenthesis in expression")
                raise ValueError("Mismatched parentheses: unmatched left parenthesis")

            while operator_stack:
                self.__apply_operator(operand_stack, operator_stack)

            result = operand_stack[0] if operand_stack else dict()
            self.__logger.info(
                f"Expression parsing completed, result: {len(result)} documents"
            )
            return result

        except Exception as e:
            self.__logger.error(f"Failed to parse expression: {str(e)}")
            raise

    def query_cache__const(self, query: str) -> Optional[List[Tuple[int, float]]]:
        """
        Query cache for specified query result with vector space model scores.

        Args:
            query: Query string to search in cache

        Returns:
            List of (document_id, cosine_similarity_score) tuples if found in cache, None otherwise
            Results are sorted by cosine similarity scores in ascending order

        Raises:
            Exception: When cache access error occurs

        Note:
            Returns reference to internal data for performance. Do not modify the returned data.
        """
        try:
            if not query:
                self.__logger.warning("Query string is empty, cannot query cache")
                return None

            if query in self.__cache:
                _, hit_count, result = self.__cache[query]
                self.__logger.debug(
                    f"Cache hit: '{query}' (hits: {hit_count}, results: {len(result)})"
                )
                return result

            self.__logger.debug(f"Cache miss: '{query}'")
            return None

        except Exception as e:
            self.__logger.error(f"Cache query error: {str(e)}")
            return None

    def __update_cache(self, query: str, result: List[Tuple[int, float]]):
        """
        Update cache using LFU-LRU strategy with vector space model scored results.

        Args:
            query: Query string as cache key
            result: Query result as list of (document_id, cosine_similarity_score) tuples

        Raises:
            Exception: When cache update error occurs
        """
        try:
            if not query:
                self.__logger.warning("Query string is empty, cannot update cache")
                return

            if len(self.__cache) >= self.__cache_size:
                self.__logger.debug(
                    f"Cache full ({len(self.__cache)}/{self.__cache_size}), applying LFU-LRU eviction"
                )

                min_hit_count = min(
                    hit_count for _, hit_count, _ in self.__cache.values()
                )
                oldest_time = float("inf")
                key_to_remove = None

                for key, (timestamp, hit_count, _) in self.__cache.items():
                    if hit_count == min_hit_count and timestamp < oldest_time:
                        oldest_time = timestamp
                        key_to_remove = key

                if key_to_remove:
                    removed_entry = self.__cache.pop(key_to_remove)
                    self.__logger.info(
                        f"Evicted from cache: '{key_to_remove}' (hits: {removed_entry[1]})"
                    )
                else:
                    self.__logger.warning("Failed to find cache entry to remove")

            current_time = int(time.time())
            self.__cache[query] = [current_time, 1, result.copy()]
            self.__logger.info(f"Added to cache: '{query}' (results: {len(result)})")

        except Exception as e:
            self.__logger.error(f"Cache update error: {str(e)}")

    def query(self, query: str) -> List[Tuple[int, float]]:
        """
        Execute boolean query and return vector space model ranked document results.

        Processes boolean expressions using the following workflow:
        1. Parses boolean expressions and extracts search content between logical operators
        2. Tokenizes search content into individual terms
        3. Applies implicit AND logic to multi-term tokenized segments
        4. Executes boolean logic queries to find relevant documents
        5. Computes document scores using vector space model with cosine similarity
        6. Returns matching documents sorted by their relevance scores

        The method supports caching for performance optimization and provides
        comprehensive logging throughout the query execution process.

        Args:
            query: Boolean query string with operators (&&, ||, !) and keywords

        Returns:
            List of (document_id, cosine_similarity_score) tuples sorted by relevance scores
            (descending order - higher scores indicate higher relevance)

        Raises:
            ValueError: When query string is invalid or empty
            Exception: When query execution error occurs
        """
        try:
            if not query:
                self.__logger.warning("Query string is empty")
                return list()

            query = query.strip()
            if not query:
                self.__logger.warning("Query string contains only whitespace")
                return list()

            self.__logger.info(f"Executing query: '{query}'")

            cached_result = self.query_cache__const(query)
            if cached_result is not None:
                try:
                    self.__cache[query][0] = int(time.time())
                    self.__cache[query][1] += 1
                    self.__logger.info(
                        f"Cache hit, returning {len(cached_result)} documents"
                    )
                    return cached_result
                except Exception as cache_error:
                    self.__logger.error(f"Failed to update cache: {str(cache_error)}")
                    return cached_result

            start_time = time.time()
            result = sorted(
                self.__parse_expression(query).items(), key=lambda x: x[1], reverse=True
            )
            execution_time = time.time() - start_time

            self.__update_cache(query, result)

            self.__logger.info(
                f"Query '{query}' completed, found {len(result)} documents, "
                f"execution time: {execution_time:.3f}s"
            )
            return result

        except Exception as e:
            self.__logger.error(f"Query execution failed: {str(e)}")
            raise

    def convert_id_to_path(self, doc_id: int) -> str:
        """
        Convert document ID to file path using the inverted index.

        This method provides a convenient way to retrieve the original file path
        from a document ID returned by query results. It's particularly useful
        for displaying human-readable file paths instead of numeric IDs.

        Args:
            doc_id (int): Document ID to convert to file path

        Returns:
            str: Absolute file path corresponding to the document ID

        Raises:
            ValueError: When document ID is not found in the inverted index
            TypeError: When doc_id is not an integer
            Exception: When inverted index query fails
        """
        try:
            if not isinstance(doc_id, int):
                self.__logger.error(f"Document ID must be integer, got: {type(doc_id)}")
                raise TypeError(f"Document ID must be integer, got: {type(doc_id)}")

            if doc_id < 0:
                self.__logger.error(f"Document ID must be non-negative, got: {doc_id}")
                raise ValueError(f"Document ID must be non-negative, got: {doc_id}")

            self.__logger.debug(f"Converting document ID {doc_id} to file path")

            doc_path = self.__inverted_index.query_document(doc_id)
            if not doc_path:
                self.__logger.error(f"Document ID {doc_id} not found in inverted index")
                raise ValueError(f"Document ID {doc_id} not found in index")

            self.__logger.debug(f"Document ID {doc_id} resolved to path: {doc_path}")
            return doc_path

        except (ValueError, TypeError):
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            self.__logger.error(f"Failed to convert ID {doc_id} to path: {str(e)}")
            raise

    def get_cache_info(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive cache statistics and performance information.

        Returns:
            Dictionary containing cache statistics:
            - cache_size: Maximum cache capacity
            - current_entries: Number of cached queries
            - usage_rate: Cache utilization percentage
            - total_hit_count: Sum of all cache hits
            - average_hit_count: Average hits per cached query
        """
        try:
            total_queries = len(self.__cache)
            if total_queries == 0:
                usage_rate = 0.0
                total_hit_count = 0
                average_hit_count = 0.0
            else:
                total_hit_count = sum(
                    hit_count for _, hit_count, _ in self.__cache.values()
                )
                average_hit_count = total_hit_count / total_queries
                usage_rate = total_queries / self.__cache_size

            return {
                "cache_size": self.__cache_size,
                "current_entries": total_queries,
                "usage_rate": usage_rate,
                "total_hit_count": total_hit_count,
                "average_hit_count": average_hit_count,
            }
        except Exception as e:
            self.__logger.error(f"Failed to get cache info: {str(e)}")
            return {}


if __name__ == "__main__":
    # Interactive test mode for Vector Space Model Boolean retrieval system
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        print("=== Vector Space Model Boolean Retrieval Interactive Test ===")
        print(
            "This system combines boolean logic with vector space model scoring for document retrieval."
        )
        print("Features:")
        print("- Automatic tokenization of search terms between logical operators")
        print("- Boolean logic queries to find relevant documents")
        print("- Vector space model scoring using cosine similarity")
        print("- Document ID to file path conversion for result analysis")
        print("- LFU-LRU cache strategy for query performance optimization")
        print("Supported operators: && (AND), || (OR), ! (NOT), () (grouping)")
        print("Results are ranked by vector space model relevance scores.\n")

        boolRetrieval = BoolRetrieval(
            {
                "docs_dir": testDir,
                "docs_extensions": testExtensions,
                "docs_extensions_exclude": testExtensionsExclude,
                "docs_encoding": None,
            },
            cache_size=10,
        )
        print(f"BoolRetrieval system initialized successfully")
        print(f"Document index built and ready for queries")

        # Sample queries to demonstrate functionality
        while True:
            print("\n" + "=" * 60)
            print("--- Vector Space Model Boolean Retrieval Menu ---")
            print("1. Execute boolean query")
            print("2. Show cache information")
            print("3. Convert document ID to file path")
            print("0. Exit")
            print("=" * 60)

            try:
                choice = input("\nEnter your choice (0-3): ").strip()

                if choice == "0":
                    print(
                        "Thank you for using Vector Space Model Boolean Retrieval System!"
                    )
                    break

                elif choice == "1":
                    print("\nBoolean Query Execution:")
                    testQuery = input("Enter boolean query: ").strip()

                    if testQuery:
                        try:
                            startTime = time.time()
                            res = boolRetrieval.query(testQuery)
                            execTime = time.time() - startTime

                            print(f"\n--- Query Results ---")
                            print(f"Query: '{testQuery}'")
                            print(f"Documents found: {len(res)}")
                            print(f"Execution time: {execTime:.4f}s")

                            if res:
                                print(
                                    f"\nRanked Results (Document_ID, Vector_Space_Score):"
                                )
                                for i, (docId, score) in enumerate(
                                    res[:10], 1
                                ):  # Show top 10
                                    try:
                                        # Convert document ID to file path for display
                                        filePath = boolRetrieval.convert_id_to_path(
                                            docId
                                        )
                                        print(
                                            f"  {i:2d}. Document {docId} ({os.path.basename(filePath)}): {score:.6f}"
                                        )
                                    except Exception as path_error:
                                        print(
                                            f"  {i:2d}. Document {docId}: {score:.6f} (path conversion failed), {str(path_error)}"
                                        )

                                # Show document IDs only for compatibility
                                docIds = [docId for docId, _ in res]
                                print(f"\nDocument IDs: {docIds}")
                            else:
                                print("No documents match the query")

                        except Exception as excpt:
                            print(f"Query error: {str(excpt)}")
                    else:
                        print("Empty query not allowed")

                elif choice == "2":
                    print("\n--- Cache Statistics ---")
                    info = boolRetrieval.get_cache_info()
                    print(f"Cache size: {info.get('cache_size', 0)}")
                    print(f"Current entries: {info.get('current_entries', 0)}")
                    print(f"Usage rate: {info.get('usage_rate', 0.0):.2%}")
                    print(f"Total hit count: {info.get('total_hit_count', 0)}")
                    print(
                        f"Average hit count per query: {info.get('average_hit_count', 0.0):.2f}"
                    )

                elif choice == "3":
                    print("\n--- Document ID to File Path Conversion ---")
                    try:
                        docIdInput = input("Enter document ID to convert: ").strip()

                        if not docIdInput:
                            print("Empty input not allowed")
                            continue

                        try:
                            docId = int(docIdInput)
                        except ValueError:
                            print(
                                f"Invalid document ID format: '{docIdInput}'. Please enter a valid integer."
                            )
                            continue

                        try:
                            filePath = boolRetrieval.convert_id_to_path(docId)
                            print(f"\n--- Conversion Result ---")
                            print(f"Document ID: {docId}")
                            print(f"File Path: {filePath}")
                            print(f"File Name: {os.path.basename(filePath)}")
                            print(f"Directory: {os.path.dirname(filePath)}")

                            # Check if file exists
                            if os.path.exists(filePath):
                                file_size = os.path.getsize(filePath)
                                print(f"File Size: {file_size} bytes")
                                print("File Status: EXISTS")
                            else:
                                print(
                                    "File Status: NOT FOUND (file may have been moved or deleted)"
                                )

                        except ValueError as ve:
                            print(f"Document ID conversion error: {str(ve)}")
                        except TypeError as te:
                            print(f"Document ID type error: {str(te)}")
                        except Exception as path_error:
                            print(
                                f"Unexpected error during conversion: {str(path_error)}"
                            )

                    except KeyboardInterrupt:
                        print("\nOperation cancelled by user")
                    except EOFError:
                        print("\nEnd of input detected")

                else:
                    print("Invalid choice. Please enter a number between 0-3.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user.")
                break
            except EOFError:
                print("\n\nEnd of input detected.")
                break
            except Exception as excpt:
                print(f"Unexpected error: {str(excpt)}")

    except Exception as excpt:
        print(
            f"Failed to initialize Vector Space Model boolean retrieval system: {str(excpt)}"
        )
        print("\nDetailed error information:")
        traceback.print_exc()
