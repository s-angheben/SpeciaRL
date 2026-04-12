"""
Custom exceptions for the application, allowing for centralized and specific error handling.
"""

class RateLimitError(Exception):
    """Raised when a downstream API rate limit is exceeded after all retries."""
    pass

class ServiceUnavailableError(Exception):
    """Raised when a downstream service (like a vLLM server) is not available."""
    pass

class ConfigurationError(Exception):
    """Raised when there is a problem with the application's configuration."""
    pass

class LLMResponseParseError(Exception):
    """Raised when the response from the LLM is malformed or cannot be parsed."""
    pass

class IncompleteLLMResponseError(LLMResponseParseError):
    """Raised when the LLM response is valid but missing expected data."""
    def __init__(self, message, missing_count=0):
        super().__init__(message)
        self.missing_count = missing_count

class CachePollTimeoutError(Exception):
    """Raised when a follower task times out waiting for a leader to write a result to the cache."""
    pass

class InvalidPromptError(Exception):
    """Raised when an invalid verifier prompt is requested."""
    def __init__(self, prompt_name: str, available_prompts: list):
        self.prompt_name = prompt_name
        self.available_prompts = available_prompts
        super().__init__(f"Invalid verifier prompt '{prompt_name}'. Available prompts: {', '.join(available_prompts)}")

class ModelNotFoundError(Exception):
    """Raised when an invalid model is requested."""
    def __init__(self, model_name: str, available_models: list):
        self.model_name = model_name
        self.available_models = available_models
        super().__init__(f"Invalid model '{model_name}'. Available models: {', '.join(available_models)}")
