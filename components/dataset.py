import bisect
import os

import av
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def read_video_pyav(video_path, start, end):
    """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
    container = av.open(video_path)
    video = container.streams.get(0)[0]
    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]
    av_timestamps.sort()
    start_id = bisect.bisect_left(av_timestamps, start)
    end_id = bisect.bisect_left(av_timestamps, end)

    if end_id - start_id < 10:
        end_id = min(len(av_timestamps) - 1, end_id + 10)
        start_id = max(0, start_id - 10)

    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(0, start_id)
    num_frames_to_sample = min(2, end_id - start_id + 1)
    indices = np.linspace(start_id, end_id, num_frames_to_sample).astype(int)

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_id:
            break
        if i >= start_id and i in indices:
            frames.append(frame)
    assert len(
        frames) == 2, f"Got {len(frames)} frames but should be 2. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class VideoLlavaDataset(Dataset):
    """
    PyTorch Dataset for VideoLlavaDataset.
    """

    def __init__(self, video_path: str, csv_file: str, num_frames: int = 8):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
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

        clip = read_video_pyav(video_path, 0, 1e+10)
        answer = sentence
        tmp_prompt = "<video>\n Translate the American Sign Language (ASL) demonstrated in the video to English text, where each frame shows ASL signs used at different time points chronologically."

        prompt = f"USER: {tmp_prompt}" \
                 f"\n ASSISTANT: Answer: {answer}"

        return prompt, clip, answer
