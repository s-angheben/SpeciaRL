from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union, Tuple

class ModelConfig(BaseModel):
    """Configuration for VLM models."""
    model_name: str = Field(description="A unique, user-defined name for this model run (e.g., 'my_custom_qwen_v1')")
    model_type: str = Field(description="The type of model to load. Supported: 'huggingface_vlm', 'openai_vlm', 'verl_vlm'")
    path_or_id: str = Field(description="Path to a local model directory or a HuggingFace model ID")
    device_map: str = Field(default="auto", description="Device mapping strategy ('auto', 'cuda', 'cpu', etc.)")
    dtype: str = Field(default="auto", description="Model dtype ('auto', 'bf16', 'float32', etc.)")
    model_loader_class: str = Field(default="AutoModelForCausalLM", description="The specific Hugging Face class to use for loading the model.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters (e.g., api_key for OpenAI)")

class MyGenerationConfig(BaseModel):
    """Pydantic mirror of `transformers.GenerationConfig`. Unset fields fall back to the model's `generation_config.json`."""
    # Parameters that control the length of the output
    max_length: Optional[int] = Field(default=None, description="The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.")
    max_new_tokens: Optional[int] = Field(default=None, description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    min_length: Optional[int] = Field(default=None, description="The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.")
    min_new_tokens: Optional[int] = Field(default=None, description="The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    early_stopping: Optional[bool] = Field(default=None, description="Controls the stopping condition for beam-based methods. `True` means stopping as soon as `num_beams` complete candidates are found.")
    max_time: Optional[float] = Field(default=None, description="The maximum amount of time you allow the computation to run for in seconds.")
    stop_sequences: Optional[Union[str, List[str]]] = Field(default=None, description="A string or a list of strings that should terminate generation if the model outputs them. Renamed from `stop_strings` for clarity.")

    # Parameters that control the generation strategy used
    do_sample: Optional[bool] = Field(default=None, description="Whether or not to use sampling; use greedy decoding otherwise.")
    num_beams: Optional[int] = Field(default=None, description="Number of beams for beam search. 1 means no beam search.")
    num_beam_groups: Optional[int] = Field(default=None, description="Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.")
    penalty_alpha: Optional[float] = Field(default=None, description="The values balance the model confidence and the degeneration penalty in contrastive search decoding.")
    dola_layers: Optional[Union[str, List[int]]] = Field(default=None, description="The layers to use for DoLa decoding (e.g., 'low', 'high', or a list of layer indices).")

    # Parameters that control the cache
    use_cache: Optional[bool] = Field(default=None, description="Whether or not the model should use the past last key/values attentions to speed up decoding.")
    cache_implementation: Optional[str] = Field(default=None, description="The cache implementation to use (e.g., 'static', 'dynamic').")
    cache_config: Optional[Dict[str, Any]] = Field(default=None, description="A dictionary of arguments for the chosen cache implementation.")

    # Parameters for manipulation of the model output logits
    temperature: Optional[float] = Field(default=None, ge=0.0, description="The value used to module the next token probabilities. Must be >= 0.0.")
    top_k: Optional[int] = Field(default=None, gt=0, description="The number of highest probability vocabulary tokens to keep for top-k-filtering. Must be > 0.")
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.")
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum token probability, which will be scaled by the probability of the most likely token.")
    typical_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next.")
    epsilon_cutoff: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled.")
    eta_cutoff: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Eta sampling is a hybrid of locally typical sampling and epsilon sampling.")
    diversity_penalty: Optional[float] = Field(default=None, description="This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time.")
    repetition_penalty: Optional[float] = Field(default=None, description="The parameter for repetition penalty. 1.0 means no penalty.")
    encoder_repetition_penalty: Optional[float] = Field(default=None, description="The parameter for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input.")
    length_penalty: Optional[float] = Field(default=None, description="Exponential penalty to the length that is used with beam-based generation.")
    no_repeat_ngram_size: Optional[int] = Field(default=None, description="If set to int > 0, all ngrams of that size can only occur once.")
    bad_words_ids: Optional[List[List[int]]] = Field(default=None, description="List of list of token ids that are not allowed to be generated.")
    force_words_ids: Optional[Union[List[List[int]], List[List[List[int]]]]] = Field(default=None, description="List of token ids that must be generated.")
    renormalize_logits: Optional[bool] = Field(default=None, description="Whether to renormalize the logits after applying all the logits processors.")
    forced_bos_token_id: Optional[int] = Field(default=None, description="The id of the token to force as the first generated token after the `decoder_start_token_id`.")
    forced_eos_token_id: Optional[Union[int, List[int]]] = Field(default=None, description="The id of the token to force as the last generated token when `max_length` is reached.")
    remove_invalid_values: Optional[bool] = Field(default=None, description="Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.")
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = Field(default=None, description="A tuple of (start_index, decay_factor) to add an exponentially increasing length penalty.")
    suppress_tokens: Optional[List[int]] = Field(default=None, description="A list of tokens that will be suppressed at generation.")
    begin_suppress_tokens: Optional[List[int]] = Field(default=None, description="A list of tokens that will be suppressed at the beginning of the generation.")
    guidance_scale: Optional[float] = Field(default=None, description="The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.")
    watermarking_config: Optional[Dict[str, Any]] = Field(default=None, description="A dictionary of arguments for watermarking.")

    # Parameters that define the output variables of generate
    num_return_sequences: Optional[int] = Field(default=None, description="The number of independently computed returned sequences for each element in the batch.")
    output_attentions: Optional[bool] = Field(default=None, description="Whether or not to return the attentions tensors of all attention layers.")
    output_hidden_states: Optional[bool] = Field(default=None, description="Whether or not to return the hidden states of all layers.")
    output_scores: Optional[bool] = Field(default=None, description="Whether or not to return the prediction scores.")
    output_logits: Optional[bool] = Field(default=None, description="Whether or not to return the unprocessed prediction logit scores.")
    return_dict_in_generate: Optional[bool] = Field(default=None, description="Whether or not to return a `ModelOutput`.")

    # Special tokens that can be used at generation time
    pad_token_id: Optional[int] = Field(default=None, description="The id of the *padding* token.")
    bos_token_id: Optional[int] = Field(default=None, description="The id of the *beginning-of-sequence* token.")
    eos_token_id: Optional[Union[int, List[int]]] = Field(default=None, description="The id of the *end-of-sequence* token.")

    # Generation parameters exclusive to encoder-decoder models
    encoder_no_repeat_ngram_size: Optional[int] = Field(default=None, description="If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.")
    decoder_start_token_id: Optional[int] = Field(default=None, description="If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.")

    def get_user_overrides(self) -> Dict[str, Any]:
        """Return only user-set fields, renaming `stop_sequences` to the transformers `stop_strings` kwarg."""
        overrides = self.model_dump(exclude_unset=True)

        if "stop_sequences" in overrides:
            overrides["stop_strings"] = overrides.pop("stop_sequences")

        return overrides

class PredictionConfig(BaseModel):
    model: ModelConfig = Field(description="Model configuration")
    generation_params: MyGenerationConfig = Field(default_factory=MyGenerationConfig, description="Text generation parameters")
    output_dir: str = Field(default="predictions", description="Directory to save predictions")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_predictions_per_sample: int = Field(default=1, description="Number of predictions per sample")
    batch_size: int = Field(default=1, description="Batch size for predictions when num_predictions_per_sample > 1")
    inference_batch_size: int = Field(default=16, description="Number of samples to process together for better GPU utilization")
    prediction_chunk_size: int = Field(default=1, description="Maximum number of predictions per sample to generate in a single vLLM call to prevent OOMs with large 'n'.")
    target_splits: Optional[List[str]] = Field(default=None, description="List of dataset splits to run predictions on.")
    prompts_override_file: Optional[str] = Field(default=None, description="Path to prompt NDJSON file to override dataset prompts at prediction time")