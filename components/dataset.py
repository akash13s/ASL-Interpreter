import bisect
import os

import av
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


# def read_video_pyav(video_path, start, end):
#     """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
#     container = av.open(video_path)
#     video = container.streams.get(0)[0]
#     av_timestamps = [
#         int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
#     ]
#     av_timestamps.sort()
#     start_id = bisect.bisect_left(av_timestamps, start)
#     end_id = bisect.bisect_left(av_timestamps, end)

#     if end_id - start_id < 10:
#         end_id = min(len(av_timestamps) - 1, end_id + 10)
#         start_id = max(0, start_id - 10)

#     end_id = min(len(av_timestamps) - 1, end_id)
#     start_id = max(0, start_id)
#     num_frames_to_sample = min(2, end_id - start_id + 1)
#     indices = np.linspace(start_id, end_id, num_frames_to_sample).astype(int)

#     frames = []
#     container.seek(0)
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_id:
#             break
#         if i >= start_id and i in indices:
#             frames.append(frame)
#     assert len(
#         frames) == 2, f"Got {len(frames)} frames but should be 2. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


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

class VideoLlavaDataset(Dataset):
    """
    PyTorch Dataset for VideoLlavaDataset.
    """

    def __init__(self, video_path: str, csv_file: str, num_frames: int = 8, mode: str = "train"):
        super().__init__()
        df = pd.read_csv(csv_file)
        
        self.annotations = df
        if len(self.annotations) > 10000 and mode == "train":
            self.annotations = df.head(10000)
        if len(self.annotations) > 10000 and mode == "val":
            self.annotations = df.iloc[10000:10500]
        
        self.num_frames = num_frames
        self.video_path = video_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        sample = self.annotations.iloc[idx]

        # Lazy load video clip here
        video_id = str(sample['SENTENCE_NAME']).strip()
        sentence = str(sample['SENTENCE']).strip()

        video_path = os.path.join(self.video_path, f'{video_id}.mp4')

        clip = get_frames(video_path, self.num_frames)
        answer = sentence
        tmp_prompt = "<video>\n Translate the American Sign Language (ASL) demonstrated in the video to English text, where each frame shows ASL signs used at different time points chronologically."

        prompt = f"USER: {tmp_prompt}" \
                 f"\n ASSISTANT: Answer: {answer}"

        return prompt, clip, answer
