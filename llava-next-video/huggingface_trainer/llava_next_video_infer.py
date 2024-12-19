import csv
import os

import av
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig

# Constants
CACHE_DIR = "./cache/"
MODEL_CHECKPOINT = "./output/model"
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "../../data/valid_clips.csv"
OUTPUT_FILE = "./output/inference_results.csv"

DATASET_SIZE = 1

# Model constants
BATCH_SIZE = 5
MAX_LENGTH = 3500  # Fixed sequence length for text
NUM_FRAMES = 16  # Fixed number of frames
IMAGE_SIZE = 224  # Fixed image size

# Quantization parameters
USE_QLORA = True
USE_4BIT = True  # Keep false if not using QLORA
USE_8BIT = False  # Keep false if not using QLORA
USE_DBL_QUANT = True  # Keep false if not using QLORA


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


def run_inference_with_dataset(video_dir, csv_file, output_file, processor, model, num_frames, device):
    """
    Run inference using the VideoDataset and save results to a CSV file.

    Args:
        video_dir (str): Path to the video directory.
        csv_file (str): Path to the annotations CSV file.
        output_file (str): Path to save the inference results (CSV file).
        processor: Pretrained processor for the model.
        model: Loaded model for inference.
        num_frames (int): Number of frames to extract from each video.
        device: cpu or gpu.
    """
    # Load the annotations
    annotations = pd.read_csv(csv_file, sep=',').head(DATASET_SIZE).reset_index(drop=True)
    infer_dataset = VideoDataset(video_dir, annotations, processor, num_frames, "infer")

    # Check if the CSV file exists
    file_exists = os.path.exists(output_file)

    # Open the CSV file for writing
    with open(output_file, 'a', newline="") as csvfile:
        fieldnames = ["id", "video_id", "generated", "true"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file doesn't already exist
        if not file_exists:
            writer.writeheader()

        # Inference loop
        print("Starting inference...")
        for idx in range(len(infer_dataset)):
            infer_data = infer_dataset[idx]
            # Move inputs to the appropriate device
            input_ids = infer_data["input_ids"].to(device).unsqueeze(0)
            attention_mask = infer_data["attention_mask"].to(device).unsqueeze(0)
            pixel_values_videos = infer_data["pixel_values_videos"].to(device).unsqueeze(0)

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
            )
            generated_texts = processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Clean generated texts and write to CSV
            for _, text in enumerate(generated_texts):
                keyword = "ASSISTANT:"
                if keyword in text:
                    text = text.split(keyword, 1)[1].strip()

                writer.writerow({
                    "id": idx,
                    "video_id": infer_data['video_id'],
                    "generated": text,
                    "true": infer_data['true_sentence']
                })

    print(f"Inference complete. Results saved to {output_file}")


def main():
    # Load the model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.do_rescale = False
    processor.video_processor.do_rescale = False

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, MODEL_CHECKPOINT)
    model.to(device)
    model.eval()

    print("Model and processor loaded successfully.")

    run_inference_with_dataset(
        video_dir=VIDEO_DIR,
        csv_file=CSV_FILE,
        output_file=OUTPUT_FILE,
        processor=processor,
        model=model,
        num_frames=NUM_FRAMES,
        device=device
    )


if __name__ == "__main__":
    main()
