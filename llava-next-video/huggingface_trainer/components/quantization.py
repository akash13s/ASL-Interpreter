import torch
from transformers import BitsAndBytesConfig


def get_quantization_config(use_qlora: bool, use_4bit: bool, use_8bit: bool):
    """
    Generate the appropriate BitsAndBytesConfig for quantization.

    Args:
        use_qlora (bool): Whether QLoRA-specific settings should be used.
        use_4bit (bool): Enable 4-bit quantization.
        use_8bit (bool): Enable 8-bit quantization.

    Returns:
        BitsAndBytesConfig: Configured object for the quantization setup.
    """
    # Validation to avoid conflicting quantization options
    assert not (use_8bit and use_4bit), "Cannot use both 8-bit and 4-bit quantization simultaneously."

    # Base configuration
    quantization_config = {
        "load_in_8bit": use_8bit,
        "load_in_4bit": use_4bit,
        "bnb_4bit_compute_dtype": torch.float16
    }

    # Add QLoRA-specific options if enabled
    if use_qlora:
        quantization_config.update({
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        })

    return BitsAndBytesConfig(**quantization_config)
