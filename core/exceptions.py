"""
Custom exception classes for Hadamard-LLM.
"""

class HadamardException(Exception):
    """Base exception for all errors raised by Hadamard-LLM."""
    pass

class ModelProviderError(HadamardException):
    """Raised when an external LLM provider API returns an error or fails."""
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider} Error {status_code}]: {message}")

class MissingAPIKeyError(HadamardException):
    """Raised when a required API key is missing from environment variables or configuration."""
    pass

class PromptTemplateError(HadamardException):
    """Raised when prompt rendering fails due to missing or invalid variables."""
    pass
