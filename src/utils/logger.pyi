"""
Logger module for managing logging across the application.

This module provides a centralized logging setup function that creates and configures loggers for different modules. Configuration parameters are loaded from config.py including log format, level, rotation settings, and output destinations.
"""

import logging

def setup_logger(
    module_name: str,
    level_str: str = ...,
    log_format: str = ...,
    max_bytes: int = ...,
    backup_count: int = ...,
    log_dir: str = ...,
    to_console: bool = ...,
) -> logging.Logger:
    """
    Configure and return a logger for the specified module.

    Args:
        module_name (str): Name of the module for which to set up the logger.
        level_str (str): Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format (str): Format string for log messages.
        max_bytes (int): Maximum size of log file before rotation.
        backup_count (int): Number of backup files to keep.
        log_dir (str): Directory to store log files, None to disable file logging.
        to_console (bool): Whether to output logs to console.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If invalid logging level is provided.
    """
    ...
