"""
Boolean retrieval implementation for information retrieval.

This module provides boolean query processing capabilities including:
- Support for AND (&&), OR (||), and NOT (!) operators
- Parentheses for grouping expressions
- LFU-LRU cache strategy for query optimization
- Expression parsing and evaluation
"""

import sys
import os
from typing import Dict, List, Set, Tuple, Union
import re
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger
from InfoRetrieval.InvertedIndex import InvertedIndex


class BoolRetrieval:
    __logger = setup_logger(__name__)

    def __init__(
        self, identifier: Union[InvertedIndex, Dict[str, str]], cache_size: int = 100
    ):
        """
        Initialize Boolean retrieval system.

        Args:
            identifier: InvertedIndex instance or dict containing document directory path
            cache_size: Cache size for query results, default 100

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
            self.__cache: Dict[str, Tuple[int, int, Set[int]]] = {}
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
                f"Documents: {self.__inverted_index.get_docs_count()}, "
                f"Keywords: {self.__inverted_index.get_keywords_count()}"
            )

        except Exception as e:
            self.__logger.error(f"BoolRetrieval initialization failed: {str(e)}")
            raise

    def __apply_operator(
        self, operand_stack: List[Set[int]], operator_stack: List[str]
    ):
        """
        Apply top operator from operator stack.

        Args:
            operand_stack: Stack of operands (document ID sets)
            operator_stack: Stack of operators

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
                all_docs = set(range(1, self.__inverted_index.get_docs_count() + 1))
                result = all_docs.difference(operand)
                operand_stack.append(result)
                self.__logger.debug(f"NOT operation result: {len(result)} documents")

            elif op == "&&":
                if len(operand_stack) < 2:
                    self.__logger.error(
                        f"AND operation requires two operands, got: {len(operand_stack)}"
                    )
                    raise IndexError("AND operation requires two operands")

                right = operand_stack.pop()
                left = operand_stack.pop()
                result = left.intersection(right)
                operand_stack.append(result)
                self.__logger.debug(f"AND operation result: {len(result)} documents")

            elif op == "||":
                if len(operand_stack) < 2:
                    self.__logger.error(
                        f"OR operation requires two operands, got: {len(operand_stack)}"
                    )
                    raise IndexError("OR operation requires two operands")

                right = operand_stack.pop()
                left = operand_stack.pop()
                result = left.union(right)
                operand_stack.append(result)
                self.__logger.debug(f"OR operation result: {len(result)} documents")

            else:
                self.__logger.error(f"Unknown operator: {op}")
                raise ValueError(f"Unsupported operator: {op}")

        except Exception as e:
            self.__logger.error(f"Failed to apply operator: {str(e)}")
            raise

    def __parse_expression(self, expr: str) -> Set[int]:
        """
        Parse boolean expression and compute results using stacks.
        Operator precedence: ! > && > ||

        Args:
            expr: Boolean expression string

        Returns:
            Set of document IDs matching the condition

        Raises:
            ValueError: When expression format is invalid
            Exception: When parsing errors occur
        """
        try:
            if not expr or not expr.strip():
                self.__logger.warning("Query expression is empty")
                return set()

            self.__logger.info(f"Parsing expression: {expr}")

            tokens = re.findall(r"(?:[^\\&|!()]|\\.)+|&&|\|\||!|\(|\)", expr)
            self.__logger.debug(f"Tokenization result: {tokens}")

            if not tokens:
                self.__logger.warning("Expression tokenization resulted in empty list")
                return set()

            operand_stack = []
            operator_stack = []
            precedence = {"(": 0, "||": 1, "&&": 2, "!": 3}

            i = 0
            while i < len(tokens):
                token = tokens[i].strip()
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
                            i += 1
                            continue

                        keyword = re.sub(r"\\([&|!()])", r"\1", token)
                        self.__logger.debug(f"Processing keyword: {keyword}")

                        postings = self.__inverted_index.get_postings__const(keyword)
                        if postings is not None:
                            doc_set = postings[1].copy()
                            operand_stack.append(doc_set)
                            self.__logger.debug(
                                f"Keyword '{keyword}' found in {len(doc_set)} documents"
                            )
                        else:
                            operand_stack.append(set())
                            self.__logger.debug(
                                f"Keyword '{keyword}' not found in any documents"
                            )

                    i += 1

                except Exception as e:
                    self.__logger.error(f"Error processing token '{token}': {str(e)}")
                    raise

            if "(" in operator_stack:
                self.__logger.error("Unmatched left parenthesis in expression")
                raise ValueError("Mismatched parentheses: unmatched left parenthesis")

            while operator_stack:
                self.__apply_operator(operand_stack, operator_stack)

            result = operand_stack[0] if operand_stack else set()
            self.__logger.info(
                f"Expression parsing completed, result: {len(result)} documents"
            )
            return result

        except Exception as e:
            self.__logger.error(f"Failed to parse expression: {str(e)}")
            raise

    def query_cache(self, query: str) -> Union[Set[int], None]:
        """
        Query cache for specified query result.

        Args:
            query: Query string

        Returns:
            Document ID set if found in cache, None otherwise

        Raises:
            Exception: When cache access error occurs
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
                return result.copy()

            self.__logger.debug(f"Cache miss: '{query}'")
            return None

        except Exception as e:
            self.__logger.error(f"Cache query error: {str(e)}")
            return None

    def __update_cache(self, query: str, result: Set[int]):
        """
        Update cache using LFU-LRU strategy.

        Args:
            query: Query string
            result: Query result

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
            self.__cache[query] = (current_time, 1, result.copy())
            self.__logger.info(f"Added to cache: '{query}' (results: {len(result)})")

        except Exception as e:
            self.__logger.error(f"Cache update error: {str(e)}")

    def query(self, query: str) -> Set[int]:
        """
        Execute boolean query and return matching document IDs.

        Args:
            query: Query string

        Returns:
            Set of document IDs matching the condition

        Raises:
            ValueError: When query string is invalid
            Exception: When query execution error occurs
        """
        try:
            if not query:
                self.__logger.warning("Query string is empty")
                return set()

            query = query.strip()
            if not query:
                self.__logger.warning("Query string contains only whitespace")
                return set()

            self.__logger.info(f"Executing query: '{query}'")

            cached_result = self.query_cache(query)
            if cached_result is not None:
                try:
                    timestamp, hit_count, result = self.__cache[query]
                    self.__cache[query] = (timestamp, hit_count + 1, result)
                    self.__logger.info(
                        f"Cache hit, returning {len(cached_result)} documents"
                    )
                    return cached_result
                except Exception as cache_error:
                    self.__logger.error(
                        f"Failed to update cache hit count: {str(cache_error)}"
                    )
                    return cached_result

            start_time = time.time()
            result = self.__parse_expression(query)
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

    def get_cache_info(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics information.

        Returns:
            Dictionary containing cache statistics
        """
        try:
            total_queries = len(self.__cache)
            if total_queries == 0:
                return {
                    "cache_size": self.__cache_size,
                    "current_entries": 0,
                    "usage_rate": 0.0,
                    "total_hit_count": 0,
                    "average_hit_count": 0.0,
                }

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
    # Interactive test mode for boolean retrieval
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        print("=== Boolean Retrieval Interactive Test ===")
        boolRetrieval = BoolRetrieval(
            {
                "docs_dir": testDir,
                "docs_extensions": testExtensions,
                "docs_extensions_exclude": testExtensionsExclude,
                "docs_encoding": None,
            },
            cache_size=10,
        )
        print(f"BoolRetrieval initialized successfully")

        while True:
            print("\n--- Menu ---")
            print("1. Execute boolean query")
            print("2. Show cache information")
            print("3. Clear cache")
            print("0. Exit")

            try:
                choice = input("\nEnter your choice (0-3): ").strip()

                if choice == "0":
                    print("Goodbye!")
                    break

                elif choice == "1":
                    testQuery = input("Enter boolean query: ").strip()
                    if testQuery:
                        try:
                            startTime = time.time()
                            res = boolRetrieval.query(testQuery)
                            execTime = time.time() - startTime

                            print(f"Query: '{testQuery}'")
                            print(f"Results: {len(res)} documents found")
                            print(f"Execution time: {execTime:.4f}s")

                            if res:
                                print(f"Document IDs: {sorted(list(res))}")
                            else:
                                print("No documents match the query")

                        except Exception as excpt:
                            print(f"Query error: {str(excpt)}")
                    else:
                        print("Empty query not allowed")

                elif choice == "2":
                    info = boolRetrieval.get_cache_info()
                    print("Cache Statistics:")
                    print(f"\tCache size: {info.get('cache_size', 0)}")
                    print(f"\tCurrent entries: {info.get('current_entries', 0)}")
                    print(f"\tUsage rate: {info.get('usage_rate', 0.0):.2%}")
                    print(f"\tTotal hit count: {info.get('total_hit_count', 0)}")
                    print(
                        f"  Average hit count: {info.get('average_hit_count', 0.0):.2f}"
                    )

                elif choice == "3":
                    print("Cache cleared.")

                else:
                    print("Invalid choice. Please enter a number between 0-3.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user.")
                break
            except Exception as excpt:
                print(f"Error: {str(excpt)}")

    except Exception as excpt:
        print(f"Failed to initialize boolean retrieval: {str(excpt)}")
        import traceback

        traceback.print_exc()
