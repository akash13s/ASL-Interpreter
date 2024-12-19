import av
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoProcessor


def read_video_pyav(container: av.container.input.InputContainer, indices: np.ndarray, image_size: int) -> np.ndarray:
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`np.ndarray`): List of frame indices to decode.
        image_size (`int`): Image size.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
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


def get_frames(video_path: str, num_frames: int, image_size: int) -> np.ndarray:
    """
    Extract frames from video with consistent sampling
    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract
        image_size (int): Image size
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
    frames = read_video_pyav(container, indices, image_size)

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


def get_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.do_rescale = False
    processor.video_processor.do_rescale = False
    processor.patch_size = 14  # Standard patch size for ViT-L
    return processor


def generate_text(model, processor, sample, device=torch.device("cpu")):
    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
    pixel_values_videos = sample['pixel_values_videos'].unsqueeze(0).to(device)

    # Generate predictions
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )
    generated_text = processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Clean the generated text
    keyword = "ASSISTANT:"
    if keyword in generated_text:
        generated_text = generated_text.split(keyword, 1)[1].strip()

    return generated_text
