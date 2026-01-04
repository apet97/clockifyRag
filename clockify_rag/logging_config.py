"""Centralized logging configuration for Clockify RAG system.

This module provides a single point of configuration for all logging
across the application, ensuring consistent formatting, levels, and handlers.

Includes support for:
- JSON and text formatters
- Console and file handlers
- Log rotation for file handlers
- Dedicated query log with rotation
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available (set by CorrelationIdFilter)
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id and correlation_id != "-":
            log_obj["correlation_id"] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_obj["extra"] = record.extra

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with colors (if supported)."""

    # Color codes (ANSI)
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True, include_correlation_id: bool = True):
        """Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes
            include_correlation_id: Whether to include correlation ID in output
        """
        self.include_correlation_id = include_correlation_id
        # Base format - correlation ID will be prepended if available
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors and correlation ID."""
        original_levelname = record.levelname
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{original_levelname}{reset}"

        try:
            result = super().format(record)
        finally:
            if self.use_colors:
                record.levelname = original_levelname

        # Prepend correlation ID if available
        if self.include_correlation_id:
            correlation_id = getattr(record, "correlation_id", None)
            if correlation_id and correlation_id != "-":
                # Show short form (first 8 chars) for readability
                short_id = correlation_id[:8] if len(correlation_id) > 8 else correlation_id
                result = f"[{short_id}] {result}"

        return result


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    quiet: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5,
    use_rotation: bool = True,
    enable_correlation_ids: bool = True,
) -> None:
    """Central logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("text" or "json")
        log_file: Optional file path for file logging
        use_colors: Use colored output for console (text mode only)
        quiet: Suppress console output (only log to file)
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        use_rotation: Whether to use RotatingFileHandler (default True)
        enable_correlation_ids: Whether to include correlation IDs in logs (default True)

    Example:
        >>> # Development mode
        >>> setup_logging(level="DEBUG", format_type="text", use_colors=True)
        >>>
        >>> # Production mode with rotation
        >>> setup_logging(level="INFO", format_type="json", log_file="/var/log/rag.log",
        ...               max_bytes=50*1024*1024, backup_count=10)
        >>>
        >>> # File-only logging
        >>> setup_logging(level="INFO", log_file="app.log", quiet=True)
    """
    from .correlation import CorrelationIdFilter

    # Get root logger
    root = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    root.handlers.clear()

    # Set level
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"Warning: Invalid log level '{level}', using INFO", file=sys.stderr)

    root.setLevel(log_level)

    # Create correlation ID filter (adds correlation_id to all log records)
    correlation_filter = CorrelationIdFilter() if enable_correlation_ids else None

    # Create formatter based on type
    formatter: logging.Formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(use_colors=use_colors, include_correlation_id=enable_correlation_ids)

    # Console handler (unless quiet mode)
    if not quiet:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        if correlation_filter:
            console.addFilter(correlation_filter)
        root.addHandler(console)

    # File handler (if specified)
    if log_file:
        # Create parent directories if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler for automatic log rotation
        if use_rotation:
            file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                log_file,
                mode="a",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        file_handler.setLevel(log_level)

        # Always use JSON for file logs for easier parsing
        file_handler.setFormatter(JSONFormatter())
        if correlation_filter:
            file_handler.addFilter(correlation_filter)
        root.addHandler(file_handler)

    # Set third-party library log levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Log configuration applied
    rotation_info = f"rotation={use_rotation}, max_bytes={max_bytes}, backups={backup_count}" if log_file else ""
    root.info(
        f"Logging configured: level={level}, format={format_type}, file={log_file or 'none'}, quiet={quiet}, correlation_ids={enable_correlation_ids}, {rotation_info}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)


# Convenience function for testing
def reset_logging() -> None:
    """Reset logging configuration.

    Useful for testing to ensure clean state between tests.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# Module-level query logger singleton
_QUERY_LOGGER: Optional[logging.Logger] = None
_QUERY_LOGGER_PATH: Optional[str] = None  # Track current log file path
_QUERY_LOGGER_LOCK = None


def get_query_logger(
    log_file: str = "rag_queries.jsonl",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5,
) -> logging.Logger:
    """Get or create a dedicated query logger with rotation.

    This logger is specifically for query audit logging, writing
    JSONL entries to a rotating file. It's separate from the main
    application logging to allow independent configuration.

    If the log_file path changes, the logger is automatically recreated
    with the new path (useful for testing with temp directories).

    Args:
        log_file: Path to query log file
        max_bytes: Maximum file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)

    Returns:
        Logger instance configured for query logging

    Example:
        >>> logger = get_query_logger("rag_queries.jsonl")
        >>> logger.info(json.dumps({"query": "...", "answer": "..."}))
    """
    import threading

    global _QUERY_LOGGER, _QUERY_LOGGER_PATH, _QUERY_LOGGER_LOCK

    # Lazy initialize lock
    if _QUERY_LOGGER_LOCK is None:
        _QUERY_LOGGER_LOCK = threading.Lock()

    # Normalize path for comparison
    normalized_path = str(Path(log_file).resolve())

    with _QUERY_LOGGER_LOCK:
        # If logger exists but path changed, reset it
        if _QUERY_LOGGER is not None and _QUERY_LOGGER_PATH != normalized_path:
            for handler in _QUERY_LOGGER.handlers:
                handler.flush()
                handler.close()
            _QUERY_LOGGER.handlers.clear()
            _QUERY_LOGGER = None
            _QUERY_LOGGER_PATH = None

        if _QUERY_LOGGER is not None:
            return _QUERY_LOGGER

        # Create dedicated logger for queries (doesn't propagate to root)
        query_logger = logging.getLogger("clockify_rag.queries")
        query_logger.setLevel(logging.INFO)
        query_logger.propagate = False  # Don't send to root logger

        # Clear any existing handlers
        query_logger.handlers.clear()

        # Create parent directories if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler for automatic log rotation
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

        # Simple formatter that just outputs the message (already JSON)
        handler.setFormatter(logging.Formatter("%(message)s"))
        query_logger.addHandler(handler)

        _QUERY_LOGGER = query_logger
        _QUERY_LOGGER_PATH = normalized_path
        return _QUERY_LOGGER


def reset_query_logger() -> None:
    """Reset the query logger singleton.

    Useful for testing to ensure clean state.
    """
    global _QUERY_LOGGER, _QUERY_LOGGER_PATH
    if _QUERY_LOGGER is not None:
        # Flush and close all handlers before clearing
        for handler in _QUERY_LOGGER.handlers:
            handler.flush()
            handler.close()
        _QUERY_LOGGER.handlers.clear()
        _QUERY_LOGGER = None
        _QUERY_LOGGER_PATH = None


def flush_query_logger() -> None:
    """Flush the query logger to ensure all writes are persisted.

    Useful for testing or before process exit.
    """
    global _QUERY_LOGGER
    if _QUERY_LOGGER is not None:
        for handler in _QUERY_LOGGER.handlers:
            handler.flush()
