import av
import bisect
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import json
import torch.nn.utils.prune as prune
from tqdm import tqdm
import os
from typing import Tuple, Any
import pandas as pd
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

# Constants
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# File/directory
VIDEO_DIR = "/scratch/as18464/raw_videos"
CSV_FILE = "valid_clips.csv"
CACHE_DIR = "cache/"
DATASET_SIZE = 10000

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
BATCH_SIZE = 8
MAX_LENGTH = 350

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, video_dir: str, csv_file: str, num_frames: int = 8):
        self.video_dir = video_dir
        self.annotations = pd.read_csv(csv_file, sep=',').head(DATASET_SIZE).reset_index(drop=True)
        self.num_frames = num_frames
        print(f"Loaded dataset with {len(self.annotations)} entries")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        row = self.annotations.iloc[idx]
        video_id = str(row['SENTENCE_NAME']).strip()
        sentence = str(row['SENTENCE']).strip()

        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")

        frames = get_frames(video_path, self.num_frames)

        tmp_prompt = "Translate the American Sign Language (ASL) demonstrated in the video to English text, where each frame shows ASL signs used at different time points chronologically."

        prompt = f"USER: <video> {tmp_prompt}\nASSISTANT: Answer: {sentence}"

        frames_list = [frame for frame in frames]
        frames_list = [transforms.ToTensor()(frame) for frame in frames_list]
        frame_tensor = torch.stack(frames_list)

        return prompt, frame_tensor

def train_epoch(model, train_loader, optimizer, processor, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')


    for batch_idx, (texts, videos) in enumerate(progress_bar):
        vids = list(torch.unbind(videos, dim=0))
        image_lists = []
        for batch in vids:
            images = [img.permute(1, 2, 0).numpy() for img in batch]
            image_lists.append(images)
        try:
            batch = processor(
                text=texts,
                videos=image_lists,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values_videos = batch["pixel_values_videos"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                labels=labels
            )
            loss = outputs.loss

            if isinstance(model, DataParallel):
                loss = loss.mean()

            loss.backward()
            # torch.nn.utils.clip_gradnorm(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({
                'batch_loss': f'{current_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })

            # Print detailed loss information for each iteration
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {current_loss:.4f} | Avg Loss: {avg_loss:.4f}')

        except Exception as e:
            raise e

    # Save checkpoint after every epoch
    checkpoint_path = f"output/checkpoint_epoch_{epoch}"
    os.makedirs("output", exist_ok=True)  # Ensure output directory exists

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }

    print(f"Saving checkpoint for epoch {epoch} with average loss: {avg_loss:.4f}")
    torch.save(checkpoint, checkpoint_path)

    return total_loss / len(train_loader)

# Create dataset and dataloader
def create_data_loader(video_dir, csv_file, batch_size, num_frames=8):
    dataset = VideoDataset(
        video_dir=video_dir,
        csv_file=csv_file,
        num_frames=num_frames
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=False
    )

    return loader

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    quantization_config=quantization_config
)

train_loader = create_data_loader(
    video_dir=VIDEO_DIR,
    csv_file=CSV_FILE,
    batch_size=BATCH_SIZE,
    num_frames=16
)

p_model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Get PEFT model
p_model = get_peft_model(p_model, peft_config)
p_model.print_trainable_parameters()

p_model = p_model.cuda()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    p_model = DataParallel(p_model)

optimizer = torch.optim.AdamW(p_model.parameters(), lr=1e-3)
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
processor.image_processor.do_rescale = False

for i in range(4):
    train_epoch(p_model, train_loader, optimizer, processor, device, i)