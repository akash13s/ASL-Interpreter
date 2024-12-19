import os

import pandas as pd
import torch
from datasets import Dataset

from components.model import get_model
from components.peft import load_peft_model
from components.preprocessor import Preprocessor
from components.quantization import get_quantization_config
from components.utils import get_processor, generate_text
from constants import *

# Clean-up tasks
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
os.makedirs(f"{CACHE_DIR}", exist_ok=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Load Dataset
df = pd.read_csv(f"{OUTPUT_DIR}/{CSV_FILE}")
max_epoch = df['epoch'].max()
df = df[df['epoch'] == max_epoch]
dataset = Dataset.from_pandas(df)

# Load Quantization Config
quantization_config = get_quantization_config(USE_QLORA, USE_4BIT, USE_8BIT)

# Load Model
model = get_model(MODEL_ID, CACHE_DIR, quantization_config)

# Create lora_params
lora_params = {
    "r": LORA_R,
    "alpha": LORA_ALPHA,
    "dropout": LORA_DROPOUT,
    "target_modules": LORA_TARGET_MODULES
}

# Load Peft Model
p_model = load_peft_model(model, lora_params)

# Load Processor
processor = get_processor(MODEL_ID)

# Load Pre-processor
preprocessor = Preprocessor(VIDEO_DIR, processor, NUM_FRAMES, IMAGE_SIZE, MAX_LENGTH, "eval")
dataset = (dataset
           .map(preprocessor, remove_columns=dataset.column_names)
           .set_format("pt", columns=["input_ids", "attention_mask", "pixel_values_videos"], output_all_columns=True))

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataset.map(lambda row: {"generated": generate_text(model, processor, row, device)})
