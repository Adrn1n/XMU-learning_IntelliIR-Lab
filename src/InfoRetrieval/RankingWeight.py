import sys
import os
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Utils.Logger import setup_logger


class RankingWeightCalculator:
    __logger = setup_logger(__name__)

    @classmethod
    def tf_idf(cls, tf, df, total_docs):
        try:
            cls.__logger.debug(
                f"Calculating TF-IDF: tf={tf}, df={df}, total_docs={total_docs}"
            )

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
    def cosine_similarity(cls, vector_a, vector_b):
        try:
            cls.__logger.debug(
                f"Calculating cosine similarity: vector_a={vector_a}, vector_b={vector_b}"
            )

            if isinstance(vector_a, list):
                vector_a = np.array(vector_a)

            if isinstance(vector_b, list):
                vector_b = np.array(vector_b)

            if vector_a.shape != vector_b.shape:
                cls.__logger.warning(
                    "Input vectors must have the same shape for cosine similarity calculation."
                )
                return 0.0
            elif vector_a.size == 0 or vector_b.size == 0:
                cls.__logger.warning(
                    "Input vectors must not be empty for cosine similarity calculation."
                )
                return 0.0

            norm_a = float(np.linalg.norm(vector_a))
            norm_b = float(np.linalg.norm(vector_b))

            if norm_a == 0 or norm_b == 0:
                cls.__logger.warning(
                    "Zero magnitude vector detected. Cosine similarity is undefined."
                )
                return 0.0

            similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

            return similarity

        except Exception as e:
            cls.__logger.error(
                f"Unexpected error in cosine similarity calculation: {str(e)}"
            )
            return 0.0


if __name__ == "__main__":
    # Test the RankingWeightCalculator functionality
    print("=== Information Retrieval Calculator Test ===")
    print("Choose an option:")
    print("1. TF-IDF ranking weight calculation")
    print("2. Cosine similarity calculation")
    print("3. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                print("\n--- TF-IDF Calculation ---")
                print("Enter TF-IDF parameters (or '0' to return to main menu)")

                while True:
                    try:
                        tfInput = input("Enter term frequency (tf): ").strip()

                        if tfInput == "0":
                            break

                        dfInput = input("Enter document frequency (df): ").strip()

                        if dfInput == "0":
                            break

                        totalDocsInput = input(
                            "Enter total number of documents: "
                        ).strip()

                        if totalDocsInput == "0":
                            break

                        result = RankingWeightCalculator.tf_idf(
                            int(tfInput), int(dfInput), int(totalDocsInput)
                        )
                        print(f"TF-IDF weight: {result:.6f}")
                    except ValueError:
                        print("Error: Please enter valid integers for all parameters.")
                    except Exception as excpt:
                        print(f"Error calculating TF-IDF: {excpt}")
            elif choice == "2":
                print("\n--- Cosine Similarity Calculation ---")
                print(
                    "Enter two vectors (space-separated numbers, or '0' to return to main menu)"
                )

                while True:
                    try:
                        vectorAInput = input("Enter first vector: ").strip()

                        if vectorAInput == "0":
                            break

                        vectorBInput = input("Enter second vector: ").strip()

                        if vectorBInput == "0":
                            break

                        # Parse vectors from space-separated input
                        vectorA = [float(x) for x in vectorAInput.split()]
                        vectorB = [float(x) for x in vectorBInput.split()]
                        result = RankingWeightCalculator.cosine_similarity(
                            vectorA, vectorB
                        )
                        print(f"Cosine similarity: {result:.6f}")
                    except ValueError:
                        print("Error: Please enter valid numbers separated by spaces.")
                    except Exception as excpt:
                        print(f"Error calculating cosine similarity: {excpt}")
            elif choice == "3":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
