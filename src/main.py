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
from api.OllamaIntegrate import OllamaIntegrate

from config import (
    DOCS_DIR,
    DOC_EXTENSIONS,
    DOC_EXTENSIONS_EXCLUDE,
    DOC_ENCODING,
    CACHE_SIZE,
    QUERY_FILE,
    PRINT_RESULTS,
    RESULTS_PATH,
    OLLAMA_RAG_MAX_DOCS,
    OLLAMA_ANSWER_STREAMING,
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


def boolean_query_mode():
    """
    Execute boolean query workflow with performance monitoring.
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
    queries = []
    if QUERY_FILE:
        try:
            with open(QUERY_FILE, "r", encoding="utf-8") as f:
                queries = f.readlines()
            print(f"Loaded {len(queries)} queries from {QUERY_FILE}")
        except Exception as e:
            print(f"Failed to load query file: {e}")
            return
    else:
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
            results.append(list())
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
        print("Query results with TF-IDF scores:")
        for i, result in enumerate(results, 1):
            if result:
                print(f"\tQuery {i}: {len(result)} documents found")
                for j, (docId, score) in enumerate(result):
                    print(f"\t\t{j+1}. Document {docId}: TF-IDF = {score:.6f}")
            else:
                print(f"\tQuery {i}: No documents found")

    if RESULTS_PATH and isinstance(RESULTS_PATH, str):
        try:
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                f.write(
                    "=== Information Retrieval System Results with TF-IDF Scores ===\n"
                )
                f.write(f"Construction time: {construction_time:.4f} seconds\n")
                f.write(f"Memory usage: {post_construction_memory:.2f} MB\n")
                f.write(f"Total query time: {total_query_time:.4f} seconds\n")
                f.write(
                    f"Average query time: {sum(query_times)/len(query_times):.4f} seconds\n\n"
                )

                for i, (result, query_time) in enumerate(zip(results, query_times), 1):
                    if result:
                        f.write(
                            f"Query {i}: {len(result)} documents found (time: {query_time:.4f}s)\n"
                        )
                        for j, (docId, score) in enumerate(result):  # Save all results
                            f.write(
                                f"\t{j+1}. Document {docId}: TF-IDF = {score:.6f}\n"
                            )
                    else:
                        f.write(
                            f"Query {i}: No documents found (time: {query_time:.4f}s)\n"
                        )
                    f.write("\n")
            print(f"Results saved to {RESULTS_PATH}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    print(f"\n=== Program Completed Successfully ===")
    print(f"Total execution time: {time.time() - construction_start_time:.4f} seconds")


def ollama_integrate_mode():
    """
    Execute Ollama integration mode for RAG (Retrieval-Augmented Generation).
    """
    print("=== Ollama RAG Integration Mode ===")
    print("Initializing Ollama integration system...")

    try:
        # Initialize Ollama integration
        ollama_integration = OllamaIntegrate(
            max_docs=OLLAMA_RAG_MAX_DOCS,
            identifier={
                "docs_dir": DOCS_DIR,
                "docs_extensions": DOC_EXTENSIONS,
                "docs_extensions_exclude": DOC_EXTENSIONS_EXCLUDE,
                "docs_encoding": DOC_ENCODING,
            },
            cache_size=CACHE_SIZE,
        )

        # Display available models
        models = ollama_integration.get_model_list()
        current_model = ollama_integration.get_model()

        print(f"\nAvailable Ollama models:")
        for i, model in enumerate(models, 1):
            status = "(current)" if model == current_model else ""
            print(f"\t{i}. {model}{status}")

        # Model selection
        print("\nModel Selection:")
        print("Press Enter to use current model, or enter a number to switch")
        model_input = input("Model choice: ").strip()

        if model_input:
            try:
                # Try to parse as number
                model_index = int(model_input) - 1
                if 0 <= model_index < len(models):
                    selected_model = models[model_index]
                    if ollama_integration.set_model(selected_model):
                        print(f"Switched to model: {selected_model}")
                    else:
                        print(f"Failed to switch to model: {selected_model}")
                else:
                    print(
                        f"Invalid selection. Please enter a number between 1 and {len(models)}"
                    )
            except ValueError:
                # Try to match model name directly
                if model_input in models:
                    if ollama_integration.set_model(model_input):
                        print(f"Switched to model: {model_input}")
                    else:
                        print(f"Failed to switch to model: {model_input}")
                else:
                    print(
                        f"Model '{model_input}' not found. Please use a number or exact model name."
                    )
        else:
            print(f"Using current model: {current_model}")

        print("\n=== Interactive Q&A Session ===")
        print("Commands:")
        print("\t\\new - Start new conversation")
        print("\t\\no_query - Answer without document retrieval")
        print("\t\\query{your_query} - Use custom boolean query")
        print("\t'exit' or 'quit' - Exit the session")
        print("\nEnter your questions:")

        conversation_count = 0
        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() in ["exit", "quit"]:
                    break

                if not question:
                    continue

                # Get answer from Ollama integration
                print("\nAnswer: ", end="", flush=True)

                if OLLAMA_ANSWER_STREAMING:
                    # Handle streaming mode - print each chunk as it arrives
                    complete_answer = ""
                    for chunk in ollama_integration.answer(question, stream=True):
                        print(chunk, end="", flush=True)
                        complete_answer += chunk
                    print()  # Add a newline after streaming completes
                else:
                    # Handle non-streaming mode - print the complete answer
                    response = ollama_integration.answer(question, stream=False)
                    print(response)

                conversation_count += 1

            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting Ollama integration mode...")
                break
            except Exception as e:
                print(f"Error processing question: {e}")
                continue

        print(f"\nSession completed. Total conversations: {conversation_count}")

    except ConnectionError:
        print("Error: Cannot connect to Ollama service.")
        print("Please ensure Ollama is running and accessible.")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error in Ollama integration: {e}")


def main():
    """
    Main function to provide user interface for choosing between modes.
    """
    print("=== Information Retrieval System ===")
    print("Note: Press Ctrl+D to exit the program")

    while True:
        try:
            print("\n" + "=" * 50)
            print("Please choose a mode:")
            print(
                "1. Boolean Query Mode - Traditional boolean search with TF-IDF scoring"
            )
            print(
                "2. Ollama RAG Mode - AI-powered question answering with document retrieval"
            )
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                print("\n" + "=" * 50)
                boolean_query_mode()
                print("\nReturning to main menu...")
            elif choice == "2":
                print("\n" + "=" * 50)
                ollama_integrate_mode()
                print("\nReturning to main menu...")
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
