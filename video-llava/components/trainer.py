import torch
from tqdm import tqdm
import os

def train_epoch(model, train_loader, optimizer, processor, accelerator, epoch, config):
    model.train()
    total_loss = 0
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
                    if processor.tokenizer.decode(batch["input_ids"][i][j:j+4]) == "ASSISTANT:":
                        assistant_start = j
                        break
                
                if assistant_start is not None:
                    # Mask everything before and including "ASSISTANT:"
                    labels[i, :assistant_start+4] = -100

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

    if accelerator.is_main_process and epoch%5 == 0:
        checkpoint_path = f"output/checkpoint_epoch_{epoch}"
        os.makedirs("output", exist_ok=True)
        
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