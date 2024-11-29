from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

# Default Model Params
DEVICE = 0

MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
CACHE_DIR = "./cache"

USE_QLORA = True
USE_8BIT = False  # Change to use 8bit configuration with qlora, otherwise, default is 4bit

# lora parameters
LORA_R = 64
LORA_ALPHA = 128


def get_base_model(model_id, bnb_config, cache_dir, device):
    return VideoLlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"": device},
    )


def get_bnb_config(use_qlora: bool, use_8bit: bool):
    if use_qlora:
        # QLORA setup with quantization 4bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if use_8bit:
            # #QLORA setup with quantization 8bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    else:
        # LORA setup without quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=False,
            llm_int8_threshold=0.5,  # Lower threshold for increased precision
            llm_int8_skip_modules=None,  # None if skipping is not needed
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=True,  # Use FP16 weights for better precision
            # Ensures highest precision in computations
            bnb_4bit_compute_dtype=torch.float16
        )

    return bnb_config


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_lora_config(model, lora_r, lora_alpha):
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )


def get_video_llava_peft_model(
    model_id: str = MODEL_ID,
    use_qlora: bool = USE_QLORA,
    use_8bit: bool = USE_8BIT,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    cache_dir: str = CACHE_DIR,
    device: int = DEVICE
):
    bnb_config = get_bnb_config(use_qlora, use_8bit)
    model = get_base_model(model_id, bnb_config, cache_dir, device)
    lora_config = get_lora_config(model, lora_r, lora_alpha)

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_model = get_peft_model(model, lora_config)
    return peft_model
