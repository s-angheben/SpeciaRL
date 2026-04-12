from fastapi import APIRouter, Request, HTTPException
from src.schemas import ClassificationRequest, StandardClassificationResponse, BinaryClassificationResponse
from src.services.classification_service import classification_service
from src.core.config import settings
from src.core.prompt import PROMPT_REGISTRY
from src.core.exceptions import InvalidPromptError, ModelNotFoundError
from .dependencies import get_llm_client_for_model
from .routes.runs import router as runs_router

api_router = APIRouter()

@api_router.post(
    "/classify/standard",
    response_model=StandardClassificationResponse,
    tags=["Classification"],
    summary="Classify items using the standard category set"
)
async def classify_standard(
    request: Request,
    request_data: ClassificationRequest
):
    """
    Performs classification using the standard set of categories:
    - specific
    - less specific
    - generic
    - more specific
    - wrong
    - abstain
    """
    model_name = request_data.model or settings.DEFAULT_LLM_MODEL
    prompt_name = request_data.verifier_prompt or settings.DEFAULT_VERIFIER_PROMPT
    
    if prompt_name not in PROMPT_REGISTRY:
        raise InvalidPromptError(prompt_name, list(PROMPT_REGISTRY.keys()))
    
    prompt_template = PROMPT_REGISTRY[prompt_name]
    if prompt_template.get("category_set_name") != "standard":
        raise HTTPException(
            status_code=400,
            detail=f"Prompt '{prompt_name}' is not a 'standard' classification prompt. Use the /api/v1/classify/binary endpoint for this prompt."
        )
    
    available_models = list(request.app.state.llm_client_pools.keys())
    if model_name not in available_models:
        raise ModelNotFoundError(model_name, available_models)
        
    model_client = await get_llm_client_for_model(request, model_name)
    
    classifications = await classification_service.get_classifications(
        request_data=request_data,
        model_client=model_client,
        categories=prompt_template["categories"]
    )
    return StandardClassificationResponse(classifications=classifications)

@api_router.post(
    "/classify/binary",
    response_model=BinaryClassificationResponse,
    tags=["Classification"],
    summary="Classify items as 'correct' or 'wrong' using binary classification"
)
async def classify_binary(
    request: Request,
    request_data: ClassificationRequest
):
    """
    Performs classification using binary categories:
    - correct
    - wrong
    """
    model_name = request_data.model or settings.DEFAULT_LLM_MODEL
    prompt_name = request_data.verifier_prompt or settings.DEFAULT_VERIFIER_PROMPT

    if prompt_name not in PROMPT_REGISTRY:
        raise InvalidPromptError(prompt_name, list(PROMPT_REGISTRY.keys()))

    prompt_template = PROMPT_REGISTRY[prompt_name]
    if prompt_template.get("category_set_name") != "binary":
        raise HTTPException(
            status_code=400,
            detail=f"Prompt '{prompt_name}' is not a 'binary' classification prompt. Use the /api/v1/classify/standard endpoint for this prompt."
        )

    available_models = list(request.app.state.llm_client_pools.keys())
    if model_name not in available_models:
        raise ModelNotFoundError(model_name, available_models)

    model_client = await get_llm_client_for_model(request, model_name)

    classifications = await classification_service.get_classifications(
        request_data=request_data,
        model_client=model_client,
        categories=prompt_template["categories"]
    )
    return BinaryClassificationResponse(classifications=classifications)

@api_router.get(
    "/prompts",
    tags=["Classification"]
)
async def get_available_prompts():
    """Get list of available verifier prompts"""
    return {
        "available_prompts": list(PROMPT_REGISTRY.keys()),
        "default_prompt": settings.DEFAULT_VERIFIER_PROMPT
    }

@api_router.get(
    "/models",
    tags=["Classification"]
)
async def get_available_models(request: Request):
    """Get list of available models"""
    return {
        "available_models": list(request.app.state.llm_client_pools.keys()),
        "default_model": settings.DEFAULT_LLM_MODEL
    }

@api_router.get(
    "/schema",
    tags=["Documentation"]
)
async def get_request_schema():
    """Get the request schema and examples for the classify endpoints"""
    return {
        "endpoints": [
            "POST /api/v1/classify/standard",
            "POST /api/v1/classify/binary"
        ],
        "required_fields": {
            "ground_truths": {
                "type": "array of strings",
                "description": "The ground truth labels for classification",
                "example": ["cat", "dog", "bird"]
            },
            "predictions": {
                "type": "array of strings", 
                "description": "The predicted labels to be classified",
                "example": ["feline", "puppy", "robin"]
            }
        },
        "optional_fields": {
            "model": {
                "type": "string",
                "description": "Model name to use for classification",
                "default": settings.DEFAULT_LLM_MODEL,
                "available_models": "See GET /api/v1/models"
            },
            "verifier_prompt": {
                "type": "string",
                "description": "Verifier prompt template to use",
                "default": settings.DEFAULT_VERIFIER_PROMPT,
                "available_prompts": "See GET /api/v1/prompts"
            }
        },
        "examples": {
            "standard": {
                "endpoint": "POST /api/v1/classify/standard",
                "body": {
                    "ground_truths": ["cat", "dog"],
                    "predictions": ["feline", "canine"],
                    "verifier_prompt": "ver_base_json"
                }
            },
            "binary": {
                "endpoint": "POST /api/v1/classify/binary",
                "body": {
                    "ground_truths": ["cat", "dog"],
                    "predictions": ["feline", "canine"],
                    "verifier_prompt": "ver_base_json_binary"
                }
            }
        }
    }

api_router.include_router(runs_router)