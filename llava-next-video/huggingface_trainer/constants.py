# Constants
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# Model constants
BATCH_SIZE = 5
MAX_LENGTH = 3500  # Fixed sequence length for text
NUM_FRAMES = 16  # Fixed number of frames
IMAGE_SIZE = 224  # Fixed image size

# Quantization parameters
USE_QLORA = True
USE_4BIT = False  # Keep false if not using QLORA
USE_8BIT = True  # Keep false if not using QLORA

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

# File/Directory Constants
CACHE_DIR = "../cache/"
OUTPUT_DIR = "../output/"
CSV_FILE = "generated_texts_300.csv"
VIDEO_DIR = "/scratch/as18464/raw_videos"