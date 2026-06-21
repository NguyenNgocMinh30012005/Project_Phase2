from __future__ import annotations


class TryOnError(Exception):
    """Base exception for recoverable try-on failures."""


class InputValidationError(TryOnError):
    """Raised when user input is missing or incompatible."""


class ModelUnavailableError(TryOnError):
    """Raised when a required model checkpoint or dependency is missing."""


class EngineExecutionError(TryOnError):
    """Raised when a configured engine fails during execution."""
