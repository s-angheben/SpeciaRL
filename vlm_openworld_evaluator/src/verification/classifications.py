from enum import Enum

class VerificationStatus(str, Enum):
    """Represents the high-level outcome of a verification attempt."""
    SUCCESS = "success"
    PREPARATION_FAILURE = "preparation_failure"
    API_FAILURE = "api_failure"

class VerificationErrorType(str, Enum):
    """Provides a specific reason for a verification failure."""
    # Preparation Errors
    PARSING_GROUND_TRUTH = "parsing_ground_truth"
    PARSING_ANSWER_FORMAT = "parsing_answer_format"
    PARSING_VLM_OUTPUT = "parsing_vlm_output"
    PREDICTION_TOO_LONG = "prediction_too_long"
    
    # API Errors
    API_NETWORK_ERROR = "api_network_error"
    API_RESPONSE_ERROR = "api_response_error"
    API_UNKNOWN_ERROR = "api_unknown_error"