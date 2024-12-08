import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from components.pre_processor import get_frames
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, video_dir: str, csv_file: str, dataset_size: int, num_frames: int = 8):
        self.video_dir = video_dir
        self.annotations = pd.read_csv(csv_file, sep=',').head(dataset_size).reset_index(drop=True)
        self.num_frames = num_frames
        self.system_prompt = ("Analyze the American Sign Language (ASL) signs in this video and "
                              "translate them into clear, natural English. Consider the sequence of "
                              "signs as a complete message, and provide an accurate translation that "
                              "captures the full meaning. Respond with only the English translation, "
                              "without descriptions of the signs themselves.")

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

        prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT: {sentence}"

        frames_list = [frame for frame in frames]
        frames_list = [transforms.ToTensor()(frame) for frame in frames_list]
        frame_tensor = torch.stack(frames_list)

        return prompt, frame_tensor
