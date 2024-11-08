import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import av
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):

    def __init__(self, video_dir, csv_file, transform=None, fps=30):
        self.video_dir = video_dir
        self.annotations = pd.read_csv(csv_file, sep='\t')
        self.transform = transform
        self.fps = fps
        self.max_frames = self.calculate_max_frames()


    def calculate_max_frames(self):
        durations = self.annotations['END_REALIGNED'] - self.annotations['START_REALIGNED']
        max_duration = durations.max()
        return int(max_duration * self.fps)


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.annotations.iloc[idx]
        video_id = row['SENTENCE_NAME']
        sentence = row['SENTENCE']
        
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        if not os.path.isfile(video_path):
            print(f"Warning: Video file '{video_path}' not found for VIDEO_ID '{video_id}'. Returning zero-padded frames.")
            return self._get_zero_padded_frames(), sentence
        
        frames = self._get_frames(video_path)

        return frames, sentence


    def _get_frames(self, video_path):
        frames = self._load_video_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [transforms.ToTensor()(frame) for frame in frames]
        frames = self._pad_frames(frames)
        return torch.stack(frames)


    def _load_video_frames(self, video_path):
        frames = []

        container = av.open(video_path)

        for frame in container.decode(video=0):
            frame = frame.to_image()
            frames.append(frame)

        container.close()
        return frames


    def _pad_frames(self, frames):        
        num_frames = len(frames)
        if num_frames < self.max_frames:
            padding = [torch.zeros_like(frames[0])] * (self.max_frames - num_frames)
            frames.extend(padding)

        return frames

    def _get_zero_padded_frames(self):
        return torch.zeros((self.max_frames, 3, 224, 224))

def getDefaultTransform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])