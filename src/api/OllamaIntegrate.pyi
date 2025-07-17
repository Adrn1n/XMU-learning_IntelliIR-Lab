"""
Ollama Integration Module for RAG (Retrieval-Augmented Generation) System.

This module provides integration with Ollama language models to create a complete RAG system that combines document retrieval with AI-generated responses. It supports conversational interactions, boolean query generation, and document-based question answering with flexible query processing.

Features:
- Integration with Ollama language models
- Automatic boolean query generation from natural language questions
- Document retrieval using boolean search with TF-IDF scoring
- Conversational context management with history tracking
- Flexible query processing with special commands
- Document limit configuration for response optimization
- Error handling and comprehensive logging

Special Query Commands:
- \\new: Start a new conversation (clear history)
- \\no_query: Answer without document retrieval
- \\query{custom_query}: Use custom boolean search query

Dependencies:
- ollama: Ollama Python client for model interaction
- BoolRetrieval: Boolean search system with TF-IDF scoring
- FileLoader: Document loading with encoding detection
- Logger: Centralized logging system
"""

import os
import sys
import logging
from typing import Dict, List, Union, Optional, Generator

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from InfoRetrieval.BoolRetrieval import BoolRetrieval

class OllamaIntegrate:
    """
    RAG (Retrieval-Augmented Generation) system integrating Ollama models with document retrieval.

    This class provides a complete RAG implementation that combines:
    - Ollama language models for natural language processing
    - Boolean retrieval system with TF-IDF scoring for document search
    - Conversational context management for multi-turn interactions
    - Flexible query processing with special command support

    The system workflow:
    1. User asks a question in natural language
    2. System generates boolean query from question (optional)
    3. Documents are retrieved using boolean search
    4. Ollama model generates answer based on question and retrieved documents
    5. Response is returned with conversation history maintained

    Attributes:
        __model_list (List[str]): Available Ollama models
        __model (str): Currently selected Ollama model
        __max_docs (int): Maximum documents to retrieve per query
        __bool_retriever (BoolRetrieval): Boolean search system instance
        __conversation_history (List[Dict]): Conversation context history
    """

    __logger: logging.Logger
    __model_list: List[str]
    __model: str
    __max_docs: Optional[int]
    __bool_retriever: BoolRetrieval
    __conversation_history: List[Dict[str, str]]

    def __init__(self, max_docs: Optional[int] = None, **kwargs) -> None:
        """
        Initialize Ollama RAG integration system.

        Args:
            max_docs (Optional[int]): Maximum number of documents to retrieve per query
            **kwargs: Additional arguments for boolean retriever configuration

        Raises:
            RuntimeError: If Ollama is not accessible or initialization fails
        """
        ...

    def __load_models(self) -> List[str]:
        """
        Load available Ollama models from the server.

        Returns:
            List[str]: List of available model names
        """
        ...

    def get_model_list(self) -> List[str]:
        """
        Get a copy of the available Ollama models list.

        Returns:
            List[str]: Copy of available model names to prevent external modification
        """
        ...

    def get_model(self) -> str:
        """
        Get the currently selected Ollama model.

        Returns:
            str: Name of currently selected model, empty string if no model is set
        """
        ...

    def set_model(self, model_name: str) -> bool:
        """
        Set the Ollama model to use for generation.

        Args:
            model_name (str): Name of the model to set

        Returns:
            bool: True if model was set successfully, False otherwise
        """
        ...

    def set_max_docs(self, max_docs: int) -> int:
        """
        Set maximum number of documents to retrieve per query.

        Args:
            max_docs (int): Maximum number of documents

        Returns:
            int: The set maximum document count
        """
        ...

    def __generate_bool_query_from_question(
        self,
        question: str,
        sys_prompt: str = ...,
        **kwargs,
    ) -> str:
        """
        Generate boolean query from natural language question using Ollama.

        Args:
            question (str): Natural language question
            sys_prompt (str): System prompt for query generation
            **kwargs: Additional arguments for Ollama generation

        Returns:
            str: Generated boolean query string
        """
        ...

    def __generate_query_docs_response(self, query: str) -> List[str]:
        """
        Retrieve documents using boolean query.

        Args:
            query (str): Boolean query string

        Returns:
            List[str]: List of document contents
        """
        ...

    def __create_streaming_generator(
        self,
        response_chunks=...,
        content_only: bool = ...,
        update_history: bool = ...,
    ) -> Generator[Union[str, Dict[str, str]], None, None]:
        """
        Create streaming response generator from Ollama response chunks.

        Args:
            response_chunks: Ollama response chunks
            content_only (bool): Whether to return only content or full response
            update_history (bool): Whether to update conversation history

        Yields:
            Union[str, Dict[str, str]]: Response content or full response dict
        """
        ...

    def answer_question(
        self,
        question: str,
        query: Optional[Union[str, bool]] = ...,
        sys_prompt: str = ...,
        is_new: bool = ...,
        stream: bool = ...,
        **kwargs,
    ) -> Union[Dict[str, Union[str, List[str]]], Generator[Dict[str, str], None, None]]:
        """
        Answer question using RAG (Retrieval-Augmented Generation).

        Args:
            question (str): Question to answer
            query (Optional[Union[str, bool]]): Boolean query or True for auto-generation
            sys_prompt (str): System prompt for answer generation
            is_new (bool): Whether to start new conversation
            stream (bool): Whether to stream response
            **kwargs: Additional arguments for Ollama generation

        Returns:
            Union[Dict, Generator]: Response dict or streaming generator
        """
        ...

    def answer(
        self,
        input_text: str,
        stream: bool = ...,
        **kwargs,
    ) -> Union[Dict[str, Union[str, List[str]]], Generator[Dict[str, str], None, None]]:
        """
        Process input text with special command handling for RAG system.

        This method processes the input text for special commands and then calls answer_question.
        Supports special commands:
        - \\new: Start a new conversation (clear history)
        - \\no_query: Answer without document retrieval
        - \\query{custom_query}: Use custom boolean search query

        Args:
            input_text (str): User input text with optional special commands
            stream (bool): Whether to stream response
            **kwargs: Additional arguments for answer_question and Ollama generation

        Returns:
            Union[Dict, Generator]: Response dict or streaming generator
        """
        ...
