import sys
import os
from polyglot.text import Text

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger


class Tokenizer:
    __logger = setup_logger(__name__)

    @staticmethod
    def tokenize(text, language="auto"):
        if not text or not text.strip():
            return []

        return Tokenizer._polyglot_tokenizer(text, language)

    @classmethod
    def _polyglot_tokenizer(cls, text, language="auto"):
        try:
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
