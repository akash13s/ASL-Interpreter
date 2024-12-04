import sys
from components.dataset import VideoLlavaDataset
from components.model import get_video_llava_peft_model
from components.collate import Collator
from transformers import AutoProcessor
from components.lightning import VideoLlavaModelPLModule
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader


# Model Constants
DEVICE = 0

MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]
CACHE_DIR = "./cache"

USE_QLORA = True
USE_8BIT = False

LORA_R = 64
LORA_ALPHA = 128

MAX_LENGTH = 350

# Data Directories
train_csv_file = "./data/valid_clips.csv"
train_video_dir = "/scratch/as18464/raw_videos"

val_csv_file = "./data/valid_clips.csv"
val_video_dir = "/scratch/as18464/raw_videos"


# Load Datasets
train_dataset = VideoLlavaDataset(video_path=train_video_dir, csv_file=train_csv_file, mode="train")
val_dataset = VideoLlavaDataset(video_path=val_video_dir, csv_file=val_csv_file, mode="val")


# Load Processor
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right


# Load Collate Functions
train_collate_fn = Collator(processor, is_val=False, max_length=MAX_LENGTH)
val_collate_fn = Collator(processor, is_val=True, max_length=MAX_LENGTH)


# Setup LVLM Model
torch.cuda.empty_cache()
model = get_video_llava_peft_model(
    model_id=MODEL_ID,
    use_qlora=USE_QLORA,
    use_8bit=USE_8BIT,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    cache_dir=CACHE_DIR,
    device=DEVICE
)



# training constants
BATCH_SIZE = 2

lora_type = "QLORA" if USE_QLORA else "LORA"
bit_type = "8bit" if USE_8BIT else "4bit"

MODEL_PATH = f"./outputs/{MODEL_NAME}_{lora_type}_{bit_type}_r{LORA_R}_alpha{LORA_ALPHA}/"


# training config
config = {
    "max_epochs": 5,
    "val_check_interval": 0.2, # how many times we want to validate during an epoch
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,
    "lr": 1e-4,
    "batch_size": BATCH_SIZE,
    "num_nodes": 1,
    "warmup_steps": 50,
    "max_new_tokens": MAX_LENGTH,
    "num_workers": 2
}


# Load Lightning Training Module
model_module = VideoLlavaModelPLModule(
    config=config,
    processor=processor,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    train_collate_fn=train_collate_fn,
    val_collate_fn=val_collate_fn
)
early_stop_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min")


# Define checkpoint callback to save only the most recent 2 checkpoints
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,  # Keeps only the best 2 checkpoints
    monitor="train_loss",  # Monitor training loss for checkpointing
    mode="min",  # Minimize the train_loss
    save_last=True,  # Always save the latest checkpoint
    dirpath=MODEL_PATH,  # Path to save the checkpoints
    filename="videollava-{epoch:02d}-{train_loss:.2f}"  # Checkpoint file naming convention
)


trainer = Trainer(
    default_root_dir=MODEL_PATH,
    accelerator="gpu",
    devices=[DEVICE],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    limit_val_batches=1,
    num_sanity_val_steps=1,
    callbacks=[early_stop_callback, checkpoint_callback],  # Add checkpoint callback here
    log_every_n_steps=1
)



# Run Trainer
trainer.fit(model_module)


# Save the processor and model locally
processor.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)