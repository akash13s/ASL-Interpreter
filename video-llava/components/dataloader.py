# Create dataset and dataloader
from torch.utils.data import DataLoader

from dataset import VideoDataset


def create_data_loader(
        video_dir: str,
        csv_file: str,
        batch_size: int,
        dataset_size: int,
        num_frames: int = 8
):
    dataset = VideoDataset(
        video_dir=video_dir,
        csv_file=csv_file,
        dataset_size=dataset_size,
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
