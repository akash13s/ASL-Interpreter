import os

import torch
from tqdm import tqdm


def train_epoch(config, epoch):
    loss = None
    total_loss = 0
    avg_loss = 0

    model = config['model']
    optimizer = config['optimizer']
    train_loader = config['train_loader']
    processor = config['processor']
    accelerator = config['accelerator']
    output_dir = config['output_dir']

    model.train()
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')

    for batch_idx, (texts, videos) in enumerate(progress_bar):
        vids = list(torch.unbind(videos, dim=0))
        image_lists = []
        for batch in vids:
            images = [img.cpu().permute(1, 2, 0).numpy() for img in batch]
            image_lists.append(images)
        try:
            batch = processor(
                text=texts,
                videos=image_lists,
                padding=True,
                truncation=True,
                max_length=config.get('max_length'),
                return_tensors="pt"
            )

            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            for i, text in enumerate(texts):
                assistant_start = None
                # Look for sequence: "ASSISTANT:"
                for j in range(len(batch["input_ids"][i])):
                    if processor.tokenizer.decode(batch["input_ids"][i][j:j + 4]) == "ASSISTANT:":
                        assistant_start = j
                        break

                if assistant_start is not None:
                    # Mask everything before and including "ASSISTANT:"
                    labels[i, :assistant_start + 4] = -100

            # To remove later - for debugging
            # print("\n====== Tokens and Labels for Batch", batch_idx, "======")
            # for i, text in enumerate(texts):
            #     print(f"\nOriginal text {i}: {text}")
            #     print("\nTokens and their labels:")
            #     tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
            #     for j, (token, label) in enumerate(zip(tokens, labels[i])):
            #         print(f"Position {j:3d} | Token: {token:15} | Label: {label.item():5}")
            #     print("-" * 50)

            batch["labels"] = labels

            input_ids = accelerator.prepare(batch["input_ids"])
            attention_mask = accelerator.prepare(batch["attention_mask"])
            pixel_values_videos = accelerator.prepare(batch["pixel_values_videos"])
            labels = accelerator.prepare(batch["labels"])

            input_ids = input_ids.to(accelerator.device)
            n_video_tokens = (input_ids == processor.tokenizer.convert_tokens_to_ids("<video>")).sum().item()
            frame_count = pixel_values_videos.shape[1]
            height, width = pixel_values_videos.shape[3], pixel_values_videos.shape[4]
            expected_tokens = frame_count * (height // processor.patch_size) * (width // processor.patch_size) // 4

            print(f"input_ids.size(1): {input_ids.size(1)}")
            print(f"expected_tokens: {expected_tokens}")
            print(f"n_video_tokens: {n_video_tokens}")
            print(f"Calculated dimension: {input_ids.size(1) + expected_tokens - n_video_tokens}")

            if n_video_tokens != expected_tokens:
                # Adjust attention_mask
                adjusted_attention_mask = torch.ones((1, input_ids.size(1) + expected_tokens - n_video_tokens), device=accelerator.device)
                adjusted_attention_mask[:, :input_ids.size(1)] = attention_mask
                attention_mask = adjusted_attention_mask

                # Adjust input_ids
                if n_video_tokens < expected_tokens:
                    extra_tokens = expected_tokens - n_video_tokens
                    new_tokens = torch.full((1, extra_tokens), processor.tokenizer.convert_tokens_to_ids("<video>")).to(accelerator.device)
                    input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                elif n_video_tokens > expected_tokens:
                    mask = input_ids != processor.tokenizer.convert_tokens_to_ids("<video>")
                    input_ids = input_ids[mask]
                    input_ids = input_ids[:, :expected_tokens]  # Truncate to expected length

                # Adjust labels
                adjusted_labels = torch.full_like(input_ids, -100, device=accelerator.device)  # Start with all tokens ignored
                adjusted_labels[:, :labels.size(1)] = labels
                labels = adjusted_labels

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                labels=labels
            )
            loss = outputs.loss

            accelerator.backward(loss)

            # torch.nn.utils.clip_gradnorm(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({
                'batch_loss': f'{current_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })

            if accelerator.is_main_process:
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {current_loss:.4f} | Avg Loss: {avg_loss:.4f}')

        except Exception as e:
            raise e

    if accelerator.is_main_process and epoch % 5 == 0:
        checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch}"
        os.makedirs(f"{output_dir}", exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)

        if hasattr(unwrapped_model, 'get_peft_state_dict'):
            state_dict = unwrapped_model.get_peft_state_dict()
        else:
            state_dict = unwrapped_model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        print(f"Saving checkpoint for epoch {epoch} with average loss: {avg_loss:.4f}")
        torch.save(checkpoint, checkpoint_path)

    return total_loss / len(train_loader)
