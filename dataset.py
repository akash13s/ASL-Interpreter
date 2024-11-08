import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import av
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_dir, csv_file, transform=None):
        """
        Args:
            video_dir (str): Path to the video directory
            csv_file (str): Path to the CSV file
            transform (callable, optional): Optional transform to be applied on each frame
        """
        self.video_dir = video_dir
        self.annotations = pd.read_csv(csv_file, sep='\t')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.annotations.iloc[idx]
        video_id = row['VIDEO_ID']
        video_name = row['VIDEO_NAME']
        start_time = row['START_REALIGNED']
        end_time = row['END_REALIGNED']
        sentence = row['SENTENCE']
        
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        
        if not os.path.isfile(video_path):
            print(f"Warning: Video file '{video_path}' not found for VIDEO_ID '{video_id}'. Returning empty frames.")
            empty_tensor = torch.empty((0, 3, 224, 224))
            return {"video_id": video_id, "frames": empty_tensor, "sentence": sentence}
        
        frames = self._load_video_segment(video_path, start_time, end_time)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        frames = torch.stack(frames)  # Shape: (num_frames, C, H, W)
        
        return {"video_id": video_id, "frames": frames, "sentence": sentence}
    
    
    def _load_video_segment(self, video_path, start_time, end_time):
        """Helper function to load frames from a specific segment of a video file using av."""
        frames = []
        
        container = av.open(video_path)
        
        # Seek to the start time
        container.seek(int(start_time * av.time_base))
        
        # Loop through frames until end_time is reached
        for frame in container.decode(video=0):
            if frame.time >= end_time:
                break
            # Convert frame to RGB (av frames are in YUV by default)
            frame = frame.to_image()  # Converts to PIL image in RGB
            frames.append(frame)
        
        container.close()
        return frames

def getDefaultTransform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])