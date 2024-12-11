mport av
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    VideoLlavaForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os
from typing import Tuple
import pandas as pd


# Constants
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# File/directory
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "valid_clips.csv"
CACHE_DIR = "cache/"
DATASET_SIZE = 125
TRAIN_VAL_SPLIT = 0.8

# Model constants
BATCH_SIZE = 4
MAX_LENGTH = 128  # Fixed sequence length for text
NUM_FRAMES = 16   # Fixed number of frames
IMAGE_SIZE = 224  # Fixed image size

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 20


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            # Convert to numpy array in RGB format
            frame_array = frame.to_ndarray(format="rgb24")
            # Apply resize transform and convert back to numpy
            resized_frame = resize_transform(frame_array).numpy()
            # Convert from CxHxW to HxWxC format and scale back to 0-255 range
            resized_frame = (resized_frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            frames.append(resized_frame)
    
    return np.stack(frames)

def get_frames(video_path: str, num_frames: int = 8) -> np.ndarray:
    """
    Extract frames from video with consistent sampling
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract
    Returns:
        np.ndarray: Array of frames with shape (num_frames, height, width, 3)
    """
    container = av.open(video_path)
    
    # Get video stream
    stream = container.streams.video[0]
    total_frames = stream.frames
    fps = stream.average_rate
    
    # Calculate indices to sample
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Read frames at calculated indices
    frames = read_video_pyav(container, indices)
    
    # Ensure we got exactly num_frames
    if len(frames) < num_frames:
        # If we got fewer frames, duplicate the last frame
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames = np.concatenate([frames, last_frame[np.newaxis, ...]], axis=0)
    elif len(frames) > num_frames:
        # If we got more frames, take the first num_frames
        frames = frames[:num_frames]
    
    container.close()
    return frames


class VideoDataset(Dataset):
    def __init__(self, video_dir: str, annotations: pd.DataFrame, processor, num_frames: int = 16):
        self.video_dir = video_dir
        self.annotations = annotations
        self.num_frames = num_frames
        self.processor = processor
        self.system_prompt = ("Analyze the American Sign Language (ASL) signs in this video and "
                            "translate them into clear, natural English. Consider the sequence of "
                            "signs as a complete message, and provide an accurate translation that "
                            "captures the full meaning. Respond with only the English translation, "
                            "without descriptions of the signs themselves.")
        
        print(f"Created dataset split with {len(self.annotations)} entries")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.annotations.iloc[idx]
        video_id = str(row['SENTENCE_NAME']).strip()
        sentence = str(row['SENTENCE']).strip()
        
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")
        
        # Get video frames using the provided functions
        frames = get_frames(video_path, self.num_frames)
        prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT: {sentence}"
        
        # Process the frames and text with fixed sizes
        inputs = self.processor(
            text=prompt,
            videos=[frames],  # frames is already in the correct format from get_frames
            padding="max_length",  # Always pad to max_length
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Create labels from input_ids
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Mask everything before and including "ASSISTANT:"
        assistant_start = None
        for j in range(len(inputs["input_ids"][0])):
            if self.processor.tokenizer.decode(inputs["input_ids"][0][j:j+4]) == "ASSISTANT:":
                assistant_start = j
                break
        
        if assistant_start is not None:
            labels[0, :assistant_start+4] = -100
        
        # Return tensors with consistent sizes
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def create_train_val_datasets(video_dir: str, csv_file: str, processor, num_frames: int = 16):
    # Read the full dataset
    full_df = pd.read_csv(csv_file, sep=',').head(DATASET_SIZE).reset_index(drop=True)
    
    # Calculate split sizes
    train_size = int(len(full_df) * TRAIN_VAL_SPLIT)
    val_size = len(full_df) - train_size
    
    # Randomly shuffle the dataframe
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the dataframe
    train_df = shuffled_df.iloc[:train_size]
    val_df = shuffled_df.iloc[train_size:]
    
    # Create dataset objects
    train_dataset = VideoDataset(video_dir, train_df, processor, num_frames)
    val_dataset = VideoDataset(video_dir, val_df, processor, num_frames)
    
    return train_dataset, val_dataset


def main():
    # Set up device and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.do_rescale = False

    processor.patch_size = 14  # Standard patch size for ViT-L

    # Create train and validation datasets
    train_dataset, val_dataset = create_train_val_datasets(
        video_dir=VIDEO_DIR,
        csv_file=CSV_FILE,
        processor=processor,
        num_frames=NUM_FRAMES
    )

    # Initialize model with quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        quantization_config=quantization_config
    )

    # Prepare model for k-bit training and configure LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        dataloader_num_workers=0,
    )

    # Initialize trainer without custom collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()