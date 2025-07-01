"""
File loading utilities with automatic encoding detection.

This module provides functionality to load files and directories with
automatic character encoding detection using the charset-normalizer library.
"""

import sys
import os
from typing import List, Union
from charset_normalizer import detect

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.logger import setup_logger


class FileLoader:
    """
    Utility class for loading files and directories with encoding detection.

    This class provides static methods for loading file content with automatic
    encoding detection and scanning directories for files with specific extensions.
    """

    __logger = setup_logger(__name__)

    @classmethod
    def load_file_content(
        cls, file_path: str, encoding: Union[str, None] = None
    ) -> Union[str, None]:
        """
        Load file content from the given path.

        Args:
            file_path (str): Path to the file to load.
            encoding (str, optional): Specific encoding to use. If None,
                                    encoding will be auto-detected.

        Returns:
            str: File content as string.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If encoding detection/decoding fails.
            IOError: If file cannot be read.
        """
        if not os.path.exists(file_path):
            cls.__logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if encoding:
                cls.__logger.debug(
                    f"Loading {file_path} with specified encoding {encoding}"
                )
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()

            # Auto-detect encoding
            with open(file_path, "rb") as f:
                raw_data = f.read()

            detected_encoding = detect(raw_data)["encoding"]
            if not detected_encoding:
                cls.__logger.error(f"Failed to detect encoding for {file_path}")
                raise UnicodeDecodeError(
                    "encoding_detection",
                    raw_data,
                    0,
                    len(raw_data),
                    f"Failed to detect encoding for {file_path}",
                )

            file_content = raw_data.decode(detected_encoding)
            cls.__logger.debug(
                f"Successfully loaded {file_path} with detected encoding {detected_encoding}"
            )
            return file_content

        except (IOError, OSError) as e:
            cls.__logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
        except UnicodeDecodeError as e:
            cls.__logger.error(f"Encoding error for {file_path}: {str(e)}")
            raise

    @classmethod
    def load_directory_all_files(
        cls,
        dir_path: str,
        extensions: Union[List[str], None] = None,
        extensions_exclude: Union[List[str], None] = None,
    ) -> List[str]:
        """
        Load all files with specified extensions from a directory recursively.

        Args:
            dir_path (str): Directory path to search for files.
            extensions (List[str], optional): List of file extensions to include.
                                            If None, all files are included.
            extensions_exclude (List[str], optional): List of file extensions to exclude.

        Returns:
            List[str]: List of absolute file paths found in the directory.

        Raises:
            NotADirectoryError: If dir_path is not a valid directory.
            PermissionError: If directory cannot be accessed.
        """
        if not os.path.isdir(dir_path):
            cls.__logger.error(f"Directory not found: {dir_path}")
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        abs_dir = os.path.abspath(dir_path)
        results = []

        try:
            for root, _, files in os.walk(abs_dir):
                files.sort()  # Ensure consistent ordering for testing
                for file in files:
                    ext = os.path.splitext(file)[1].lower()

                    # Check inclusion criteria
                    include_file = True
                    if extensions and ext not in extensions:
                        include_file = False
                    elif extensions_exclude and ext in extensions_exclude:
                        include_file = False

                    if include_file:
                        results.append(os.path.join(root, file))

            cls.__logger.info(f"Found {len(results)} files in {abs_dir}")
            return results

        except (OSError, PermissionError) as e:
            cls.__logger.error(f"Error accessing directory {abs_dir}: {str(e)}")
            raise


if __name__ == "__main__":
    # Test FileLoader functionality
    loader = FileLoader()

    # Test loading a directory
    testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test")
    testExtensions = [".txt", ""]
    testExtensionsExclude = [".md"]

    try:
        paths = loader.load_directory_all_files(
            testDir, testExtensions, testExtensionsExclude
        )

        print(f"Found {len(paths)} files")
        for path in paths[:5]:  # Test first 5 files
            try:
                content = loader.load_file_content(path, None)
                print(f"Loaded {path} with {len(content)} characters.")
                print(content[:100])  # Print first 100 characters
            except Exception as ex:
                print(f"Failed to load {path}: {str(ex)}")

    except Exception as excpt:
        print(f"Directory loading failed: {str(excpt)}")
