import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import VideoLlavaForConditionalGeneration

from .quantizations import get_bnb_config

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_base_model(model_id, bnb_config, cache_dir):
    return VideoLlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map="auto",
    )


def get_lora_config(lora_r, lora_alpha, lora_dropout):
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


def set_trainable_params(model):
    # First make sure all parameters are not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Then enable training only for the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:  # This targets only the LoRA layers
            param.requires_grad = True


def get_model(
        model_id: str,
        use_qlora: bool,
        use_8bit: bool,
        use_double_quant: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        cache_dir: str
):
    bnb_config = get_bnb_config(use_qlora, use_8bit, use_double_quant)
    model = get_base_model(model_id, bnb_config, cache_dir)
    lora_config = get_lora_config(lora_r, lora_alpha, lora_dropout)

    model = prepare_model_for_kbit_training(model)
    # model.gradient_checkpointing_enable()

    peft_model = get_peft_model(model, lora_config)
    set_trainable_params(peft_model)
    return peft_model
