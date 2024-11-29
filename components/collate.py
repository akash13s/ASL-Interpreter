MAX_LENGTH = 350


class Collator(object):
    def __init__(self, processor, is_val: bool, max_length=MAX_LENGTH):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.is_val = is_val

    def __call__(self, examples):
        return self.val_collate(examples) if self.is_val else self.train_collate(examples)

    def train_collate(self, examples):
        videos = []
        texts = []
        texts, videos, _ = list(zip(*examples))
        batch = self.processor(
            text=texts,
            videos=videos,
            padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = batch["input_ids"].clone()

        # We don't want to compute loss for pad tokens, lets mask with -100. Some methods also mask the prompt, calculating loss only on the answers/captions/etc
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values_videos = batch["pixel_values_videos"]
        labels = batch["labels"]

        return input_ids, attention_mask, pixel_values_videos, labels

    def val_collate(self, examples):
        # We only feed the prompt to the model
        # Make sure to separate prompt from answers/captions/etc depending on your own task and dataset
        # Otherwise your model will peek into the ground truth
        videos = []
        texts = []
        true_answers = []
        texts, videos, true_answers = list(zip(*examples))
        # Get text without answers, so the model has to generate the answers itself during eval
        texts = [text[:-2] for text in texts]
        batch = self.processor(
            text=texts,
            videos=videos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values_videos = batch["pixel_values_videos"]

        return input_ids, attention_mask, pixel_values_videos, true_answers
