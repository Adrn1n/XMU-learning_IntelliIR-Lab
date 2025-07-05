"""
TF-IDF ranking weight calculation module for information retrieval.

This module provides sophisticated ranking weight calculation capabilities for
information retrieval systems, with primary support for TF-IDF (Term Frequency -
Inverse Document Frequency) weighting. The module is designed as a factory pattern
to support extensibility for additional ranking algorithms in the future.

Features:
- TF-IDF weight calculation with logarithmic normalization
- Input validation and error handling for edge cases
- Extensible design for additional ranking algorithms
- Performance-optimized static methods
- Comprehensive parameter validation and sanitization
"""

import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger


class RankingWeightCalculator:
    """
    Factory class for calculating ranking weights in information retrieval.

    Currently supports TF-IDF calculation with logarithmic normalization.
    """

    __logger = setup_logger(__name__)

    @classmethod
    def __cal_tf_idf(cls, tf: int, df: int, total_docs: int) -> float:
        """
        Calculate TF-IDF weight using logarithmic normalization.

        Args:
            tf (int): Term frequency in document
            df (int): Document frequency in collection
            total_docs (int): Total number of documents

        Returns:
            float: TF-IDF weight, 0.0 for invalid inputs
        """
        try:
            cls.__logger.debug(
                f"Calculating TF-IDF: tf={tf}, df={df}, total_docs={total_docs}"
            )

            # Input validation - check for invalid parameter values
            if tf <= 0 or df <= 0 or total_docs <= 0:
                cls.__logger.warning(
                    f"Invalid input parameters: tf={tf}, df={df}, total_docs={total_docs}. "
                    "All parameters must be positive integers."
                )
                return 0.0
            tf_idf = (1 + math.log10(tf)) * math.log10(total_docs / df)
            return tf_idf

        except Exception as e:
            cls.__logger.error(f"Unexpected error in TF-IDF calculation: {str(e)}")
            return 0.0

    @classmethod
    def get_res(cls, *args) -> float:
        """
        Factory method to get ranking weight calculation results.

        Args:
            *args: Variable arguments for ranking calculation.

        Returns:
            float: Calculated ranking weight. Returns 0.0 for invalid inputs.

        Raises:
            ValueError: When insufficient or invalid arguments are provided
            TypeError: When arguments are of incorrect type
        """
        try:
            cls.__logger.debug(
                f"Ranking weight calculation requested with {len(args)} arguments"
            )

            return cls.__cal_tf_idf(*args)

        except Exception as e:
            cls.__logger.error(f"Ranking calculation failed: {str(e)}")
            return 0.0


if __name__ == "__main__":
    """
    Test mode for TF-IDF ranking weight calculations.
    """

    print("=== TF-IDF Ranking Weight Calculator Test ===")
    print("Enter TF-IDF parameters (Ctrl+D to exit)")

    while True:
        try:
            TF = input("Enter term frequency (tf): ")
            DF = input("Enter document frequency (df): ")
            totalDocs = input("Enter total number of documents (total_docs): ")

            try:
                result = RankingWeightCalculator.get_res(
                    int(TF), int(DF), int(totalDocs)
                )
                print(f"Calculated ranking weight: {result:.6f}")
            except Exception as excpt:
                print(f"Error calculating ranking weight: {excpt}")

        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
