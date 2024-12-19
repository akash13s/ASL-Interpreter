import os.path
import sys

import numpy as np

from .utils import get_frames


class Preprocessor:
    def __init__(self, video_dir, processor, num_frames, image_size, max_length, mode: str = 'train'):
        self.video_dir = video_dir
        if not os.path.exists(self.video_dir):
            print("Path does not exist!")
            sys.exit(1)

        self.processor = processor
        self.num_frames = num_frames
        self.mode = mode
        self.max_length = max_length
        self.image_size = image_size
        self.system_prompt = (
            "Analyze the American Sign Language (ASL) signs in this video and "
            "translate them into clear, natural English. Consider the sequence of "
            "signs as a complete message, and provide an accurate translation that "
            "captures the full meaning. Respond with only the English translation, "
            "without descriptions of the signs themselves."
        )

    def get_labels(self, inputs: dict) -> np.ndarray:
        """
        Generate label masks for training.
        """
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

    def __call__(self, example):
        """
        Preprocess a single example.
        """
        video_path = example["video_id"]
        sentence = example["true"]

        # Extract frames using get_frames
        frames = get_frames(video_path, self.num_frames, self.image_size)

        # Create the prompt
        if self.mode == "train":
            prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT: {sentence}"
        else:
            prompt = f"USER: {self.system_prompt}\n<video>\nASSISTANT:"

        # Process inputs
        inputs = self.processor(
            text=prompt,
            videos=[frames],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Prepare output dictionary
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs["pixel_values_videos"].squeeze(0),
            "video_id": example["video_id"]
        }

        if self.mode == "train":
            labels = self.get_labels(inputs)
            item["labels"] = labels.squeeze(0)

        item["true_sentence"] = sentence

        return item
