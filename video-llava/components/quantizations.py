from transformers import BitsAndBytesConfig
import torch

def get_8bit_qlora(use_double_quant: bool):
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=use_double_quant
    )

def get_4bit_qlora(use_double_quant: bool):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=use_double_quant
    )

def get_lora():
    return BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=0.5,  # Lower threshold for increased precision
        llm_int8_skip_modules=None,  # None if skipping is not needed
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=True,  # Use FP16 weights for better precision
        # Ensures highest precision in computations
        bnb_4bit_compute_dtype=torch.float16
    )

def get_bnb_config(use_qlora: bool, use_8bit: bool, use_double_quant: bool):
    if use_qlora:
        if use_8bit:
            return get_8bit_qlora(use_double_quant)
        
        return get_4bit_qlora(use_double_quant)
    
    return get_lora()