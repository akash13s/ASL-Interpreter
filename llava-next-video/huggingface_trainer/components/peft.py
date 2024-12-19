from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model


def load_peft_model(model, lora_params):
    # Prepare model for k-bit training and configure LoRA
    model = prepare_model_for_kbit_training(model)
    r, alpha, dropout, target_modules = lora_params
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    return model
