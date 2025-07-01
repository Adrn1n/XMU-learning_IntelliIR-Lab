"""
Text tokenization module for multilingual text processing.

This module provides tokenization functionality supporting multiple languages
using the Polyglot library for natural language processing tasks.
"""

import sys
import os
from typing import List
from polyglot.text import Text

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger


class Tokenizer:
    """
    Text tokenization utility class.

    Provides static methods for tokenizing text in multiple languages
    using the Polyglot natural language processing library.
    """

    __logger = setup_logger(__name__)

    @staticmethod
    def tokenize(text: str, language: str = "auto") -> List[str]:
        """
        Tokenize input text using Polyglot library.

        Args:
            text (str): Input text to be tokenized.
            language (str): Language code ('en', 'zh', 'auto' for auto-detection).

        Returns:
            List[str]: List of tokens from the input text.

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If tokenization fails.
        """
        if not text or not text.strip():
            return []

        return Tokenizer._polyglot_tokenizer(text, language)

    @classmethod
    def _polyglot_tokenizer(cls, text: str, language: str = "auto") -> List[str]:
        """
        Internal tokenization method using Polyglot library.

        Args:
            text (str): Input text to be tokenized.
            language (str): Language code for tokenization.

        Returns:
            List[str]: List of tokens.

        Raises:
            RuntimeError: If Polyglot tokenization fails.
        """
        try:
            # Create Text object with automatic or specified language detection
            if language == "auto":
                polyglot_text = Text(text)
            else:
                polyglot_text = Text(text, hint_language_code=language)

            tokens = [str(word) for word in polyglot_text.words]
            cls.__logger.debug(f"Tokenized text into {len(tokens)} tokens")
            return tokens

        except Exception as e:
            cls.__logger.error(f"Polyglot tokenization failed: {str(e)}")
            raise RuntimeError(f"Tokenization failed: {str(e)}")


if __name__ == "__main__":
    # Test tokenization functionality
    testText = input("Enter text to tokenize: ")

    try:
        res = Tokenizer.tokenize(testText)
        print(f"Tokenization result: {res}")
        print(f"Number of tokens: {len(res)}")
    except Exception as excpt:
        print(f"Tokenization failed: {excpt}")
