import os
import logging
import sys
import av
import numpy as np
import pandas as pd
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextVideoForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

# Constants
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# File/directory
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "../data/valid_clips.csv"
CACHE_DIR = "./cache/"
OUTPUT_DIR = "./output/"
LOG_DIR = "./logs"

DATASET_SIZE = 100
TRAIN_VAL_SPLIT = 0.8

# Model constants
BATCH_SIZE = 4
MAX_LENGTH = 128  # Fixed sequence length for text
NUM_FRAMES = 16  # Fixed number of frames
IMAGE_SIZE = 224  # Fixed image size

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 20

# Quantization parameters
USE_QLORA = False
USE_4BIT = False #Keep false if not using QLORA
USE_8BIT = False #Keep false if not using QLORA
USE_DBL_QUANT = False #Keep false if not using QLORA

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
    """
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
        """
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
            if self.processor.tokenizer.decode(inputs["input_ids"][0][j:j + 4]) == "ASSISTANT:":
                assistant_start = j
                break

        if assistant_start is not None:
            labels[0, :assistant_start + 4] = -100

        # Return tensors with consistent sizes
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "labels": labels.squeeze(0)
        }

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
    train_dataset = VideoDataset(video_dir, train_df, processor, num_frames)
    val_dataset = VideoDataset(video_dir, val_df, processor, num_frames)

    return train_dataset, val_dataset

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

def compute_metrics(pred, processor):
    """
    Compute BLEU, ROUGE-L, and loss for predictions.

    Args:
        pred: Prediction object from the Trainer.

    Returns:
        dict: Dictionary containing BLEU, ROUGE-L scores, and loss.
    """
    logger.info("Evaluating model after epoch...")
    predictions = pred.predictions
    labels = pred.label_ids

    # Decode predictions and labels
    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute BLEU scores
    bleu_scores = [
        sentence_bleu([label.split()], pred.split())  # Compare individual sentences
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    bleu_score = np.mean(bleu_scores)

    # Compute ROUGE scores
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [
        rouge.score(label, pred) for pred, label in zip(decoded_preds, decoded_labels)
    ]
    rouge_l = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # Include loss (Trainer logs loss automatically, so this is an example placeholder)
    # If you want loss logged, it's already available through Trainer logs.
    eval_loss = pred.metrics["eval_loss"] if "eval_loss" in pred.metrics else None

    metrics = {"bleu": bleu_score, "rouge_l": rouge_l}
    if eval_loss is not None:
        metrics["eval_loss"] = eval_loss

    logger.info(f"BLEU Score: {bleu_score}")
    logger.info(f"ROUGE-L Score: {rouge_l}")
    if eval_loss is not None:
        logger.info(f"Evaluation Loss: {eval_loss}")

    return metrics

def main():
    # Log the start of the script
    logger.info("Starting training script")

    # Set up directories
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    os.makedirs(f"{CACHE_DIR}", exist_ok=True)

    # Set up device and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.do_rescale = False
    processor.video_processor.do_rescale = False

    processor.patch_size = 14  # Standard patch size for ViT-L

    logger.info("Processor and device set up complete.")

    # Create train and validation datasets
    train_dataset, val_dataset = create_train_val_datasets(
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
        num_train_epochs=NUM_EPOCHS,
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
        metric_for_best_model="eval_loss",
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
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    logger.info("Trainer initialized. Starting training...")

    # Start training
    trainer.train()
    logger.info("Training complete.")

if __name__ == "__main__":
    main()