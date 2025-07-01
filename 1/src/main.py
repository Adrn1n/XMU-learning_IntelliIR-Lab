"""
Main program for Information Retrieval System.

This module builds a boolean query system that retrieves documents from
a document collection. It loads configuration from config.py, builds
a boolean retrieval system, processes queries, and handles results.

Workflow:
1. Load configuration from config.py
2. Build boolean retrieval system with performance monitoring
3. Get query strings (from file or input)
4. Execute boolean queries with timing
5. Process and output results with performance statistics
"""

import psutil
import os
import time
from InfoRetrieval.BoolRetrieval import BoolRetrieval

from config import (
    DOCS_DIR,
    DOC_EXTENSIONS,
    DOC_EXTENSIONS_EXCLUDE,
    DOC_ENCODING,
    CACHE_SIZE,
    QUERY_FILE,
    PRINT_RESULTS,
    RESULTS_PATH,
)


def get_memory_usage():
    """
    Get current memory usage in MB.

    Returns:
        float: Memory usage in MB
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return -1.0  # Return -1.0 if memory usage cannot be determined


def main():
    """
    Main function to execute boolean query workflow with performance monitoring.
    """
    print("=== Information Retrieval System Performance Monitor ===")

    # Record initial memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Build boolean retrieval system with timing
    print(f"\nBuilding boolean retrieval system...")
    construction_start_time = time.time()

    bool_retrieval = BoolRetrieval(
        {
            "docs_dir": DOCS_DIR,
            "docs_extensions": DOC_EXTENSIONS,
            "docs_extensions_exclude": DOC_EXTENSIONS_EXCLUDE,
            "docs_encoding": DOC_ENCODING,
        },
        CACHE_SIZE,
    )

    construction_end_time = time.time()
    construction_time = construction_end_time - construction_start_time

    # Record memory usage after construction
    post_construction_memory = get_memory_usage()
    memory_increase = post_construction_memory - initial_memory

    print(f"Boolean retrieval system built successfully")
    print(f"Construction time: {construction_time:.4f} seconds")
    print(f"Memory usage after construction: {post_construction_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")

    # Get query strings
    print(f"\nLoading queries...")
    if QUERY_FILE:
        try:
            with open(QUERY_FILE, "r", encoding="utf-8") as f:
                queries = f.readlines()
            print(f"✓ Loaded {len(queries)} queries from {QUERY_FILE}")
        except Exception as e:
            print(f"✗ Failed to load query file: {e}")
            return
    else:
        queries = []
        print("Enter queries interactively (Ctrl+C or Ctrl+D to finish):")
        while True:
            try:
                input_query = input("Enter query string: ")
                if input_query.strip():
                    queries.append(input_query)
            except (EOFError, KeyboardInterrupt):
                print(f"\nInput ended, collected {len(queries)} queries.")
                break

    if not queries:
        print("No queries to process.")
        return

    # Execute boolean queries with performance monitoring
    print(f"\nExecuting {len(queries)} queries...")
    results = []
    query_times = []
    total_query_start_time = time.time()

    for i, query in enumerate(queries, 1):
        query_stripped = query.strip()
        if not query_stripped:
            continue

        # Time individual query
        query_start_time = time.time()
        try:
            result = bool_retrieval.query(query_stripped)
            query_end_time = time.time()
            query_time = query_end_time - query_start_time

            results.append(result)
            query_times.append(query_time)

            print(
                f"\tQuery {i}: '{query_stripped}' -> {len(result)} documents ({query_time:.4f}s)"
            )

        except Exception as e:
            print(f"\tQuery {i}: '{query_stripped}' -> Error: {e}")
            results.append(set())
            query_times.append(0.0)

    total_query_end_time = time.time()
    total_query_time = total_query_end_time - total_query_start_time

    # Performance statistics
    print(f"\n=== Performance Statistics ===")
    print(f"System Construction:")
    print(f"\tConstruction time: {construction_time:.4f} seconds")
    print(
        f"\tMemory usage: {post_construction_memory:.2f} MB (increase: {memory_increase:.2f} MB)"
    )

    if query_times:
        print(f"Query Performance:")
        print(f"\tTotal queries executed: {len(query_times)}")
        print(f"\tTotal query time: {total_query_time:.4f} seconds")
        print(f"\tAverage query time: {sum(query_times)/len(query_times):.4f} seconds")
        print(f"\tFastest query: {min(query_times):.4f} seconds")
        print(f"\tSlowest query: {max(query_times):.4f} seconds")

        # Cache statistics
        cache_info = bool_retrieval.get_cache_info()
        print(f"Cache Performance:")
        print(f"\tCache size: {cache_info.get('cache_size', 0)}")
        print(f"\tCache entries: {cache_info.get('current_entries', 0)}")
        print(f"\tCache usage rate: {cache_info.get('usage_rate', 0.0):.2%}")
        print(f"\tTotal cache hits: {cache_info.get('total_hit_count', 0)}")
        print(
            f"\tAverage hits per query: {cache_info.get('average_hit_count', 0.0):.2f}"
        )

    # Final memory usage
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Total memory increase: {final_memory - initial_memory:.2f} MB")

    # Process and save results
    print(f"\n=== Results Processing ===")
    if PRINT_RESULTS:
        print("Query results:")
        for i, result in enumerate(results, 1):
            print(f"\tQuery {i}: {sorted(list(result)) if result else '[]'}")

    if RESULTS_PATH and isinstance(RESULTS_PATH, str):
        try:
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                f.write("=== Information Retrieval System Results ===\n")
                f.write(f"Construction time: {construction_time:.4f} seconds\n")
                f.write(f"Memory usage: {post_construction_memory:.2f} MB\n")
                f.write(f"Total query time: {total_query_time:.4f} seconds\n")
                f.write(
                    f"Average query time: {sum(query_times)/len(query_times):.4f} seconds\n\n"
                )

                for i, (result, query_time) in enumerate(zip(results, query_times), 1):
                    f.write(
                        f"Query {i}: {sorted(list(result)) if result else '[]'} (time: {query_time:.4f}s)\n"
                    )
            print(f"Results saved to {RESULTS_PATH}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    print(f"\n=== Program Completed Successfully ===")
    print(f"Total execution time: {time.time() - construction_start_time:.4f} seconds")


if __name__ == "__main__":
    main()
