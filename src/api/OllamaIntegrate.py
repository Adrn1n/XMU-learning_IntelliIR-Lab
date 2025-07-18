import sys
import os
import ollama
import re
from typing import Dict, List, Union
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Utils.Logger import setup_logger
from Utils.FileLoader import FileLoader
from InfoRetrieval.BoolRetrieval import BoolRetrieval

from config import (
    OLLAMA_RAG_MAX_DOCS,
    OLLAMA_SYS_PROMPT_GENERATE_BOOL_QUERY_FROM_QUESTION,
    OLLAMA_SYS_PROMPT_GENERATE_ANSWER,
    DOCS_DIR,
    DOC_EXTENSIONS,
    DOC_EXTENSIONS_EXCLUDE,
    DOC_ENCODING,
    CACHE_SIZE,
)


class OllamaIntegrate:
    __logger = setup_logger(__name__)

    def __init__(self, max_docs=None, **kwargs):
        try:
            # Load available Ollama models
            self.__model_list = self.__load_models()
            self.__model = ""
            self.__max_docs = max_docs

            # Initialize boolean retrieval system
            self.__bool_retriever: BoolRetrieval = BoolRetrieval(**kwargs)
            self.__logger.info("BoolRetrieval system initialized successfully")

            # Initialize conversation history
            self.__conversation_history = []
            self.__logger.info(
                f"OllamaIntegrate initialized with {len(self.__model_list)} models available"
            )
        except Exception as e:
            self.__logger.error(f"Failed to initialize OllamaIntegrate: {e}")
            raise RuntimeError(f"OllamaIntegrate initialization failed: {e}") from e

    def __load_models(self):
        try:
            self.__logger.debug("Attempting to load Ollama models")
            models_response = ollama.list()

            if hasattr(models_response, "models") and models_response.models:
                model_list = [model.model for model in models_response.models]

                self.__logger.info(
                    f"Successfully loaded {len(model_list)} Ollama models"
                )
                return model_list
            else:
                self.__logger.warning("No models found in Ollama service response")
                return []
        except ConnectionError as e:
            self.__logger.error(f"Failed to connect to Ollama service: {e}")
            raise ConnectionError(f"Unable to connect to Ollama service: {e}") from e
        except Exception as e:
            self.__logger.error(f"Unexpected error loading Ollama models: {e}")
            return []

    def get_model_list(self):
        return self.__model_list.copy()

    def get_model(self) -> str:
        if self.__model:
            return self.__model
        else:
            self.__logger.warning("No model has been set")
            return ""

    def set_model(self, model_name: str) -> bool:
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name must be a non-empty string")

        if model_name in self.__model_list:
            self.__model = model_name

            self.__logger.info(f"Model set to '{self.__model}'")
            return True
        else:
            self.__logger.error(
                f"Model '{model_name}' not found in available models: {self.__model_list}"
            )
            return False

    def set_max_docs(self, max_docs: int) -> int:
        if not isinstance(max_docs, int) or max_docs <= 0:
            error_msg = "max_docs must be a positive integer"
            self.__logger.error(f"Invalid max_docs value {max_docs}: {error_msg}")
            raise ValueError(error_msg)

        self.__max_docs = max_docs

        self.__logger.info(f"Maximum documents limit set to {self.__max_docs}")
        return self.__max_docs

    def __generate_bool_query_from_question(
        self,
        question,
        sys_prompt: str = OLLAMA_SYS_PROMPT_GENERATE_BOOL_QUERY_FROM_QUESTION,
        **kwargs,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not self.__model:
            raise ValueError(
                "No model is set. Please set a model before generating queries."
            )

        try:
            self.__logger.debug(
                f"Generating boolean query for question: '{question[:100]}...'"
            )
            response = ollama.generate(
                model=self.__model,
                system=sys_prompt,
                prompt=question,
                **kwargs,
            )

            # Extract and clean the response
            raw_response = response.get("response", "").strip()

            if not raw_response:
                self.__logger.warning("Empty response from Ollama model")
                return ""

            # Remove any thinking content between <think> and </think> tags
            query = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
            query = query.strip()

            self.__logger.info(
                f"Generated boolean query: '{query}' from question: '{question}'"
            )
            return query
        except Exception as e:
            self.__logger.error(
                f"Failed to generate boolean query for question '{question}': {e}"
            )
            raise RuntimeError(f"Boolean query generation failed: {e}") from e

    def __generate_query_docs_response(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            self.__logger.debug(f"Executing boolean query: '{query}'")

            # Execute boolean search query
            results = self.__bool_retriever.query(query)

            if not results:
                self.__logger.info(f"No documents found for query: '{query}'")
                return []

            self.__logger.info(
                f"Retrieved {len(results)} documents for query: '{query}'"
            )

            # Apply document limit if specified
            if self.__max_docs is not None and len(results) > self.__max_docs:
                results = results[: self.__max_docs]
                self.__logger.info(f"Limited results to {self.__max_docs} documents")

            # Convert document IDs to file paths
            document_contents = []

            for doc_id, score in results:
                try:
                    path = self.__bool_retriever.convert_id_to_path(doc_id)

                    if path:
                        content = FileLoader.load_file_content(path)

                        if content:
                            document_contents.append(content)
                            self.__logger.debug(
                                f"Loaded document {doc_id} from {path} (score: {score:.4f})"
                            )
                        else:
                            self.__logger.warning(
                                f"Empty content for document {doc_id} at {path}"
                            )
                    else:
                        self.__logger.warning(f"No path found for document ID {doc_id}")
                except Exception as e:
                    self.__logger.error(f"Failed to load document {doc_id}: {e}")
                    continue

            self.__logger.info(
                f"Successfully loaded {len(document_contents)} documents"
            )
            return document_contents
        except Exception as e:
            self.__logger.error(
                f"Failed to retrieve documents for query '{query}': {e}"
            )
            raise RuntimeError(f"Document retrieval failed: {e}") from e

    def __create_streaming_generator(
        self,
        response_chunks=None,
        content_only=False,
        update_history=True,
    ):
        # Handle case where response_chunks is None or empty (error case)
        if response_chunks is None:
            # Empty generator - yield nothing but maintain the generator protocol
            return
            # The yield statement is never reached, making this an empty generator

        complete_content = ""

        # Process streaming chunks
        for chunk in response_chunks:
            message = chunk.get("message", {})
            content = message.get("content", "")

            if content:
                complete_content += content

            # Yield based on content_only flag
            if content_only:
                if content:  # Only yield non-empty content
                    yield content
            else:
                yield chunk  # Yield full chunk

        # Add complete assistant response to conversation history if requested
        if update_history and complete_content:
            assistant_message_local = {
                "role": "assistant",
                "content": complete_content,
            }
            self.__conversation_history.append(assistant_message_local)
            self.__logger.info("Streaming answer generated successfully")
        elif update_history:
            self.__logger.warning("Empty streaming response from Ollama")

    def answer_question(
        self,
        question,
        query=None,
        sys_prompt: str = OLLAMA_SYS_PROMPT_GENERATE_ANSWER,
        is_new: bool = False,
        stream=True,
        **kwargs,
    ):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not self.__model:
            raise ValueError("Model is not set. Please set a model before querying.")

        try:
            # Clear conversation history if starting new conversation
            if is_new:
                self.__conversation_history = []
                self.__logger.debug("Started new conversation")

            # Initialize conversation with system prompt if empty
            if not self.__conversation_history:
                self.__conversation_history.append(
                    {"role": "system", "content": sys_prompt}
                )
                self.__logger.debug("Initialized conversation with system prompt")

            # Prepare the prompt with question and documents
            prompt: Dict[str, Union[str, List[str]]] = {"Question": question}

            # Handle document retrieval based on query parameter
            if query:  # Retrieve documents unless explicitly disabled
                prompt["Documents"] = []

                # Generate or use provided query
                if isinstance(query, bool):
                    # Auto-generate boolean query from question
                    generated_query = self.__generate_bool_query_from_question(question)

                    if generated_query:
                        query = generated_query
                    else:
                        self.__logger.warning(
                            "Failed to generate query, proceeding without documents"
                        )

                if isinstance(query, str) and query.strip():
                    # Retrieve documents using the query
                    doc_contents = self.__generate_query_docs_response(query)
                    prompt["Documents"] = doc_contents
                    self.__logger.info(f"Added {len(doc_contents)} documents to prompt")
                else:
                    self.__logger.info(
                        "No valid query provided, answering without documents"
                    )
            else:
                self.__logger.info("Document retrieval disabled for this question")

            # Add user message to conversation
            user_message = {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            }
            self.__conversation_history.append(user_message)
            self.__logger.debug(
                f"Generating answer with {len(self.__conversation_history)} messages in history"
            )
            kwargs["stream"] = stream  # Ensure streaming is set in kwargs
            response = ollama.chat(
                model=self.__model,
                messages=self.__conversation_history,
                **kwargs,
            )

            # Generate answer using Ollama
            if stream:
                # For streaming mode, use unified streaming generator (return full chunks)
                return self.__create_streaming_generator(
                    response, content_only=False, update_history=True
                )
            else:
                # Return complete response
                assistant_message = response.get("message", {})

                if assistant_message:
                    self.__conversation_history.append(assistant_message)
                    self.__logger.info("Answer generated successfully")

                return assistant_message
        except Exception as e:
            self.__logger.error(f"Failed to answer question '{question}': {e}")
            if stream:
                # Use the modified __create_streaming_generator with None to get an empty generator
                return self.__create_streaming_generator(
                    None, content_only=False, update_history=False
                )
            else:
                # Return empty message for non-streaming mode
                return {"role": "assistant", "content": ""}

    def answer(self, input_text, stream=True, **kwargs):
        if not input_text or not input_text.strip():
            raise ValueError("Input cannot be empty")

        try:
            # Define regex patterns for special commands
            is_new_pattern = r"(?<!\\)\\new"  # \\new command (not escaped)
            is_no_query_pattern = (
                r"(?<!\\)\\no_query"  # \\no_query command (not escaped)
            )
            query_pattern = (
                r"(?<!\\)\\query\{([^}]*)\}"  # \\query{...} command (not escaped)
            )

            # Extract command flags
            is_new = bool(re.search(is_new_pattern, input_text))
            is_no_query = bool(re.search(is_no_query_pattern, input_text))
            query_match = re.search(query_pattern, input_text)

            # Remove commands from input to get the actual question
            question = re.sub(
                f"{is_new_pattern}|{is_no_query_pattern}|{query_pattern}",
                "",
                input_text,
            ).strip()

            if not question:
                raise ValueError("Question cannot be empty after removing commands")

            # Log parsed commands
            commands_used = []

            if is_new:
                commands_used.append("\\new")
            if is_no_query:
                commands_used.append("\\no_query")
            if query_match:
                commands_used.append(f"\\query{{{query_match.group(1)}}}")

            if commands_used:
                self.__logger.debug(f"Parsed commands: {', '.join(commands_used)}")

            # Determine query parameter based on commands
            if is_no_query:
                query_param = False
            elif query_match:
                query_param = query_match.group(1)
            else:
                query_param = True  # Auto-generate query

            # Generate answer using the parsed parameters
            response = self.answer_question(
                question,
                query=query_param,
                is_new=is_new,
                stream=stream,
                **kwargs,
            )

            if stream:
                # For streaming response, use unified generator (return content only)
                return self.__create_streaming_generator(
                    response, content_only=True, update_history=False
                )
            else:
                # For non-streaming response, return content directly
                return response.get("content", "")
        except Exception as e:
            self.__logger.error(f"Failed to process input '{input_text[:100]}...': {e}")

            if stream:
                # Use the modified __create_streaming_generator with None to get an empty generator
                return self.__create_streaming_generator(
                    None, content_only=True, update_history=False
                )
            else:
                return ""  # Return empty string for non-streaming mode


if __name__ == "__main__":
    # Initialize variables that will be used in finally block
    conversation_count = 0
    ollama_integration = None
    print("=== Ollama RAG Interactive Test ===")

    # Initialize Ollama integration with error handling
    print("Initializing Ollama integration...")

    try:
        # Initialize the RAG system with configuration from config.py
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
        print("Ollama integration initialized successfully")

        # Check available models
        models = ollama_integration.get_model_list()

        if not models:
            print("Error: No Ollama models available.")
            print("Please ensure Ollama is running and models are installed.")
            print("Run 'ollama list' to check available models.")
            exit(1)
        else:
            print(f"Found {len(models)} available models")
            print(f"\nAvailable models:")
            for i, mod in enumerate(models, 1):
                print(f"\t{i}. {mod}")

        # Interactive model selection
        while True:
            try:
                choice = input(f"\nPlease select a model (1-{len(models)}): ").strip()

                if not choice.isdigit():
                    print("Please enter a valid number")
                    continue

                choice_idx = int(choice) - 1

                if not (0 <= choice_idx < len(models)):
                    print(f"Please enter a number between 1 and {len(models)}")
                    continue

                selected_model = models[choice_idx]

                if ollama_integration.set_model(selected_model):
                    print(f"✓ Model set to: {selected_model}")
                    break
                else:
                    print(f"Failed to set model: {selected_model}")
            except (ValueError, KeyboardInterrupt):
                print("\nInvalid input or operation cancelled")
                continue

        # Interactive Q&A session
        print(f"\n=== RAG Question Answering Session ===")
        print("Enter your questions below. Press Ctrl+D or Ctrl+C to exit.")
        print("\nSpecial commands:")
        print("\t\\new\t - Start a new conversation")
        print("\t\\no_query\t - Answer without document retrieval")
        print("\t\\query{...}\t - Use custom search query")
        print()

        # Reset conversation count from initialization
        conversation_count = 0

        # Main conversation loop
        while True:
            try:
                quest = input("Question: ").strip()

                if not quest:
                    print("Please enter a valid question")
                    continue

                conversation_count += 1
                print(f"\n[{conversation_count}] Processing question...")

                # Generate answer with timing
                import time

                start_time = time.time()

                # Check if user wants streaming response
                use_stream = (
                    input("Use streaming response? (Y/n): ").strip().lower() != "n"
                )

                if use_stream:
                    print(f"\nAnswer (streaming):")
                    answer_content = ""

                    for content_chunk in ollama_integration.answer(quest, stream=True):
                        print(content_chunk, end="", flush=True)
                        answer_content += content_chunk

                    print()  # New line after streaming

                    end_time = time.time()
                    response_time = end_time - start_time

                    if answer_content:
                        print(f"\n(Generated in {response_time:.2f}s)")
                        print("-" * 80)
                    else:
                        print("Failed to generate answer. Please try again.\n")
                else:
                    answer = ollama_integration.answer(quest, stream=False)
                    end_time = time.time()
                    response_time = end_time - start_time

                    if answer:
                        print(f"\nAnswer (generated in {response_time:.2f}s):")
                        print(f"{answer}\n")
                        print("-" * 80)
                    else:
                        print("Failed to generate answer. Please try again.\n")
            except EOFError:
                # Graceful exit on Ctrl+D
                break
            except KeyboardInterrupt:
                # Graceful exit on Ctrl+C
                print("\n\n\tOperation cancelled by user")
                break
            except Exception as excpt:
                print(f"Error processing question: {excpt}")
                continue
    except ConnectionError as excpt:
        print(f"Connection Error: {excpt}")
        print("Please ensure Ollama service is running.")
        print("Start Ollama with: 'ollama serve'")
    except ValueError as excpt:
        print(f"Configuration Error: {excpt}")
        print("Please check your configuration settings.")
    except Exception as excpt:
        print(f"Unexpected Error: {excpt}")
        print("Please check the logs for more details.")
    finally:
        # Session summary
        print(f"\n=== Session Summary ===")
        print(f"Total questions answered: {conversation_count}")

        if ollama_integration and ollama_integration.get_model():
            print(f"Model used: {ollama_integration.get_model()}")

        print("Thank you for using Ollama RAG!")
