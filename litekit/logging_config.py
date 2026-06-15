"""Logging configuration to send all logs to files instead of console."""

import logging
import sys
from pathlib import Path
from typing import Optional

VERBOSITY_LEVELS = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
}

def configure_logging(
    log_file: str = "litellm.log",
    level: int = logging.INFO,
    enable_console: bool = False,
    verbosity: Optional[int] = None
):
    """
    Configure logging to output to file instead of console.

    Args:
        log_file: Path to the log file
        level: Logging level (default: logging.INFO)
        enable_console: Whether to also output to console (default: False)
        verbosity: Verbosity level 0-4 (0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)
                  If provided, overrides the level parameter.
    """
    # Map verbosity level to logging level if provided
    if verbosity is not None:
        level = VERBOSITY_LEVELS.get(verbosity, logging.WARNING)

    # Ensure log file is in a 'logs' directory if no directory is specified

    # Determine the log path
    log_path = Path(log_file)
    
    # If it's just a filename or a path that doesn't look absolute,
    # put it in the root 'logs' directory
    if not log_path.is_absolute():
        # Get project root (assumed to be parent of 'litekit' directory)
        project_root = Path(__file__).parent.parent
        
        # If the path already starts with 'logs/', just make it absolute to root
        if log_path.parts[0] == "logs":
            log_path = project_root / log_path
        else:
            log_path = project_root / "logs" / log_path
            
    log_file = str(log_path)
    
    # Create logs directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    # Remove only existing stream handlers (preserve file handlers)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(level)
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    # Specifically configure LiteLLM logger
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(level)

    # Remove any stream handlers from LiteLLM logger
    for handler in litellm_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            litellm_logger.removeHandler(handler)

    # Add file handler to LiteLLM logger if it doesn't have one
    if not any(isinstance(h, logging.FileHandler) for h in litellm_logger.handlers):
        litellm_logger.addHandler(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger
