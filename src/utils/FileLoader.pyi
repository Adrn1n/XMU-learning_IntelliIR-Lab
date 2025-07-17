"""
File loading utilities with automatic encoding detection.

This module provides functionality to load files and directories with automatic character encoding detection using the charset-normalizer library.
"""

import logging
from typing import List, Optional

class FileLoader:
    """
    Utility class for loading files and directories with encoding detection.

    This class provides static methods for loading file content with automatic encoding detection and scanning directories for files with specific extensions.
    """

    __logger: logging.Logger

    @classmethod
    def load_file_content(
        cls, file_path: str, encoding: Optional[str] = None
    ) -> Optional[str]:
        """
        Load file content from the given path.

        Args:
            file_path (str): Path to the file to load.
            encoding (str, optional): Specific encoding to use. If None, encoding will be auto-detected.

        Returns:
            str: File content as string.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If encoding detection/decoding fails.
            IOError: If file cannot be read.
        """
        ...

    @classmethod
    def load_directory_all_files(
        cls,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        extensions_exclude: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Load all files with specified extensions from a directory recursively.

        Args:
            dir_path (str): Directory path to search for files.
            extensions (List[str], optional): List of file extensions to include. If None, all files are included.
            extensions_exclude (List[str], optional): List of file extensions to exclude.

        Returns:
            List[str]: List of absolute file paths found in the directory.

        Raises:
            NotADirectoryError: If dir_path is not a valid directory.
            PermissionError: If directory cannot be accessed.
        """
        ...
