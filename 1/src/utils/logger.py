"""
Logger module for managing logging across the application.

This module provides a centralized logging setup function that creates
and configures loggers for different modules. Configuration parameters
are loaded from config.py including log format, level, rotation settings,
and output destinations.
"""

import sys
import os
import logging
import logging.handlers

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_TO_CONSOLE,
    LOG_DIR,
)


def setup_logger(
    module_name: str,
    level_str: str = LOG_LEVEL or "INFO",
    log_format: str = LOG_FORMAT
    or "%(asctime)s - %(funcName)s: %(lineno)d - %(levelname)s: %(message)s",
    max_bytes: int = LOG_MAX_BYTES or 10 * 1024 * 1024,
    backup_count: int = LOG_BACKUP_COUNT or 10,
    log_dir: str = LOG_DIR or os.path.dirname(os.path.abspath(__file__)) + "../logs",
    to_console: bool = LOG_TO_CONSOLE,
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
    try:
        logger = logging.getLogger(module_name)

        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()

        # Validate and set logging level
        try:
            level = getattr(logging, level_str.upper())
            logger.setLevel(level)
        except AttributeError:
            raise ValueError(f"Invalid logging level: {level_str}")

        # Create formatter
        formatter = logging.Formatter(log_format)

        # Setup file handler if log directory is specified
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{module_name}.log")
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except (OSError, IOError) as e:
                logger.warning(
                    f"Failed to setup file logging for {module_name}: {e}. "
                    "Continuing with console logging only."
                )
                if not to_console:
                    to_console = True

        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    except Exception as e:
        fallback_logger = logging.getLogger(module_name)
        fallback_logger.setLevel(logging.INFO)
        if not fallback_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
            )
            fallback_logger.addHandler(handler)
        fallback_logger.error(f"Failed to setup logger for {module_name}: {e}")
        return fallback_logger


if __name__ == "__main__":
    # Test logger setup
    testLogger = setup_logger("test_logger")
    testLogger.info("Logger setup successful!")
    testLogger.debug("Debug message")
    testLogger.warning("Warning message")
    testLogger.error("Error message")
