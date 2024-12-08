import os

import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import AutoProcessor

from components.dataloader import create_data_loader
from components.model import get_model
from components.trainer import train_epoch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

# Constants
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# File/directory
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "../data/valid_clips.csv"
CACHE_DIR = "../cache/"
OUTPUT_DIR = "../output/"

# Quantization parameters
USE_QLORA = True
USE_8BIT = False
USE_DBL_QUANT = False

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# model constants
DATASET_SIZE = 100
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = create_data_loader(
    video_dir=VIDEO_DIR,
    csv_file=CSV_FILE,
    batch_size=BATCH_SIZE,
    dataset_size=DATASET_SIZE,
    num_frames=16
)

p_model = get_model(
    model_id=MODEL_ID,
    use_qlora=USE_QLORA,
    use_8bit=USE_8BIT,
    use_double_quant=USE_DBL_QUANT,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    cache_dir=CACHE_DIR
)

optimizer = torch.optim.AdamW(p_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# initialize the accelerator with the right kwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# Prepare your model, optimizer, and dataloader with accelerator
p_model, optimizer, train_loader = accelerator.prepare(
    p_model, optimizer, train_loader
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
processor.image_processor.do_rescale = False

config = {
    "model": p_model,
    "train_loader": train_loader,
    "optimizer": optimizer,
    "processor": processor,
    "accelerator": accelerator,
    "output_dir": OUTPUT_DIR,
    "max_length": MAX_LENGTH
}

for i in range(20):
    train_epoch(config, i + 1)
