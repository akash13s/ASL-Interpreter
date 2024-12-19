import csv
import logging
import os
import sys

import av
import numpy as np
import pandas as pd
import torch
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

# Constants
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# File/directory
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "../../data/valid_clips.csv"
CACHE_DIR = "./cache/"
OUTPUT_DIR = "./output/"
LOG_DIR = "./logs"
CHECKPOINT_PATH = "./output/checkpoint-1200"

DATASET_SIZE = 1500
TRAIN_VAL_SPLIT = 0.8

# Model constants
BATCH_SIZE = 5
MAX_LENGTH = 3500  # Fixed sequence length for text
NUM_FRAMES = 16  # Fixed number of frames
IMAGE_SIZE = 224  # Fixed image size

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 5

# Quantization parameters
USE_QLORA = True
USE_4BIT = True  # Keep false if not using QLORA
USE_8BIT = False  # Keep false if not using QLORA
USE_DBL_QUANT = True  # Keep false if not using QLORA

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

# Set up logging for both Trainer and custom logs
LOG_FILE = "./logs/training.log"
os.makedirs("./logs", exist_ok=True)  # Ensure the logs directory exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler(sys.stdout),  # Also log to stdout
    ],
)
logger = logging.getLogger(__name__)

# Capture Hugging Face Trainer logs
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
    """
    Custom Dataset for handling video data and corresponding text annotations.
    Prepares video frames and text prompts for model input.

    Args:
        video_dir (str): Directory containing the video files.
        annotations (pd.DataFrame): DataFrame containing video metadata and text annotations.
        processor: Processor for tokenizing text and preparing video frames.
        num_frames (int): Number of frames to extract from each video. Default is 16.
        mode (str): Mode of the dataset, either "train" or "eval". Default is "train".
                    If "train", the true sentence is included in the prompt. Otherwise, it is excluded.
    """

    def __init__(self, video_dir: str, annotations: pd.DataFrame, processor, num_frames: int = 16, mode: str = "train"):
        self.video_dir = video_dir
        self.annotations = annotations
        self.num_frames = num_frames
        self.processor = processor
        self.mode = mode
        self.system_prompt = ("Analyze the American Sign Language (ASL) signs in this video and "
                              "translate them into clear, natural English. Consider the sequence of "
                              "signs as a complete message, and provide an accurate translation that "
                              "captures the full meaning. Respond with only the English translation, "
                              "without descriptions of the signs themselves.")

        logger.info(f"Created dataset split with {len(self.annotations)} entries")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The length of the annotations DataFrame.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the video and text annotation for a given index, processes them into model input format.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing processed input tensors for the model:
                - input_ids: Encoded text input IDs.
                - attention_mask: Attention mask for text input.
                - pixel_values_videos: Processed video frames as tensors.
                - labels: Labels for supervised learning.
                - video_id: The ID of the video (for later use in generation and evaluation).
        """
        row = self.annotations.iloc[idx]
        video_id = str(row['SENTENCE_NAME']).strip()
        sentence = str(row['SENTENCE']).strip()

        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")

        # Get video frames using the provided functions
        frames = get_frames(video_path, self.num_frames)

        # Prepare the prompt
        if self.mode == "train":
            prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT: {sentence}"
        else:
            prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT:"  # Exclude true sentence

        # Process the frames and text with fixed sizes
        inputs = self.processor(
            text=prompt,
            videos=[frames],  # frames is already in the correct format from get_frames
            padding="max_length",  # Always pad to max_length
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        labels = None
        if self.mode == "train":
            labels = self.get_labels(inputs)

        # Return tensors with consistent sizes
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "video_id": video_id,
            "true_sentence": sentence
        }

        if self.mode == "train":
            item["labels"] = labels.squeeze(0)

        return item

    def get_labels(self, inputs: dict) -> np.ndarray:
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask everything before and including "ASSISTANT:"
        assistant_start = None
        for j in range(len(inputs["input_ids"][0])):
            if self.processor.tokenizer.decode(inputs["input_ids"][0][j:j + 4]) == "ASSISTANT:":
                assistant_start = j
                break

        if assistant_start is not None:
            labels[0, :assistant_start + 4] = -100

        return labels


def create_train_val_datasets(video_dir: str, csv_file: str, processor, num_frames: int = 16):
    """
    Creates training and validation datasets from a CSV file containing video annotations.

    Args:
        video_dir (str): Path to the directory containing video files.
        csv_file (str): Path to the CSV file containing video metadata and annotations.
        processor: Preprocessor for tokenizing text and preparing video inputs.
        num_frames (int): Number of frames to extract from each video. Default is 16.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
    """
    # Read the full dataset
    full_df = pd.read_csv(csv_file, sep=',').head(DATASET_SIZE).reset_index(drop=True)

    # Calculate split sizes
    train_size = int(len(full_df) * TRAIN_VAL_SPLIT)

    # Randomly shuffle the dataframe
    shuffled_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the dataframe
    train_df = shuffled_df.iloc[:train_size]
    val_df = shuffled_df.iloc[train_size:]

    # Create dataset objects
    train_dataset = VideoDataset(video_dir, train_df, processor, num_frames, "train")
    val_dataset = VideoDataset(video_dir, val_df, processor, num_frames, "eval")
    test_dataset = VideoDataset(video_dir, val_df, processor, num_frames, "infer")

    return train_dataset, val_dataset, test_dataset


def get_quantization_config(use_qlora: bool, use_4bit: bool, use_8bit: bool, use_double_quant: bool):
    """
    Generate the appropriate BitsAndBytesConfig for quantization.

    Args:
        use_qlora (bool): Whether QLoRA-specific settings should be used.
        use_4bit (bool): Enable 4-bit quantization.
        use_8bit (bool): Enable 8-bit quantization.
        use_double_quant (bool): Enable double quantization (QLoRA-specific).

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
            "bnb_4bit_use_double_quant": use_double_quant
        })

    return BitsAndBytesConfig(**quantization_config)


class SaveGeneratedTextsCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, output_dir):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.output_file = os.path.join(output_dir, "generated_texts.csv")

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info(f"Saving generated texts during evaluation for epoch {state.epoch}...")

        # Check if the CSV file exists
        file_exists = os.path.exists(self.output_file)

        # Open the CSV file for appending
        with open(self.output_file, 'a', newline='') as csvfile:
            fieldnames = ["epoch", "id", "video_id", "generated", "true"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header only if the file doesn't already exist
            if not file_exists:
                writer.writeheader()

            # Generate new results
            for idx in range(len(self.eval_dataset)):
                logger.info(f"Generated texts during for epoch {state.epoch}, sample {idx}...")
                sample = self.eval_dataset[idx]
                # Retrieve preprocessed inputs
                input_ids = sample['input_ids'].unsqueeze(0).to(args.device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(args.device)
                pixel_values_videos = sample['pixel_values_videos'].unsqueeze(0).to(args.device)

                # Generate predictions
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values_videos": pixel_values_videos,
                }

                model = kwargs["model"]
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
                generated_text = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # Clean the generated text
                keyword = "ASSISTANT:"
                if keyword in generated_text:
                    generated_text = generated_text.split(keyword, 1)[1].strip()

                # Write the result to the CSV file
                writer.writerow({
                    "epoch": state.epoch,
                    "id": idx,
                    "video_id": sample['video_id'],
                    "generated": generated_text,
                    "true": sample['true_sentence']
                })

        logger.info(f"Results saved to {self.output_file}.")

    def on_train_end(self, args, state, control, **kwargs):
        pass


def main():
    # Log the start of the script
    logger.info("Starting training script")

    # Set up directories
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    os.makedirs(f"{CACHE_DIR}", exist_ok=True)

    # Set up device and processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.do_rescale = False
    processor.video_processor.do_rescale = False
    processor.patch_size = 14  # Standard patch size for ViT-L

    logger.info("Processor and device set up complete.")

    # Create train and validation datasets
    train_dataset, val_dataset, test_dataset = create_train_val_datasets(
        video_dir=VIDEO_DIR,
        csv_file=CSV_FILE,
        processor=processor,
        num_frames=NUM_FRAMES
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model with quantization
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        quantization_config=get_quantization_config(USE_QLORA, USE_4BIT, USE_8BIT, USE_DBL_QUANT)
    )

    # Disable `use_cache` in the model configuration
    model.config.use_cache = False

    logger.info("Model loaded successfully.")

    # Prepare model for k-bit training and configure LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    logger.info("LoRA configuration complete.")

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS + 5,
        # resume_from_checkpoint=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        logging_dir=LOG_DIR,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_steps_per_second",  # should ideally be 'eval_loss', but 'eval_loss' value is NaN
        greater_is_better=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        dataloader_num_workers=0,
        report_to="all"
    )

    # Initialize trainer without custom collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None
    )

    # Add callback to save generated texts
    callback = SaveGeneratedTextsCallback(
        processor=processor,
        eval_dataset=test_dataset,
        output_dir=OUTPUT_DIR
    )

    trainer.add_callback(callback)

    # Start training
    if os.path.exists(CHECKPOINT_PATH):
        logger.info("Trainer initialized. Starting training from previous checkpoint...")
        trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
    else:
        logger.info("Trainer initialized. Starting training...")
        trainer.train()
    logger.info("Training complete.")

    # Save the final model and processor
    final_model_path = os.path.join(OUTPUT_DIR, "model")
    os.makedirs(final_model_path, exist_ok=True)

    # Save the trained model
    trainer.save_model(final_model_path)

    # Save the processor
    processor.save_pretrained(final_model_path)

    # Save training args
    training_args.save_to_json(os.path.join(final_model_path, "training_args.json"))

    logger.info(f"Model and processor saved to {final_model_path}")


if __name__ == "__main__":
    main()
