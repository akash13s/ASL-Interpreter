import torch
from transformers import LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig


def get_model(model_id: str, cache_directory: str, quant_config: BitsAndBytesConfig):
    # Initialize model with quantization
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_directory,
        quantization_config=quant_config
    )

    # Disable `use_cache` in the model configuration
    model.config.use_cache = False
    return model
