import torch
from torch.utils.data import DataLoader

import lightning as L


class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset, train_collate_fn, val_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.train_collate_fn = train_collate_fn
        self.val_dataset = val_dataset
        self.val_collate_fn = val_collate_fn
        self.num_workers = config.get("num_workers")
        self.max_length = config.get("max_new_tokens")
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values_videos, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=self.max_length,
            do_sample=False,
        )
        # turn them back into text, chopping of the prompt
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        correct = 0
        for pred, answer in zip(predictions, answers):
            correct += (pred.strip().lower() == answer.lower())
        self.log("val_accuracy", correct / len(answers))

        return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.val_collate_fn, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
