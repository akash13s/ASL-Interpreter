import av
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from datasets import Dataset
import torch.nn.utils.prune as prune
from tqdm import tqdm

# Constants
MAX_LENGTH = 350
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MODEL_NAME = MODEL_ID.split("/")[-1]

# Configuration
USE_LORA = False
USE_QLORA = True
USE_8BIT = False
PRUNE = True
prune_amount = 0.05
DEVICE = 4
MODEL_TYPE = "full"
batch_size = 3
lora_r = 64
lora_alpha = 128

# File paths
train_annotations = "./annotations/updated_train_annotations.json" if MODEL_TYPE == "full" else "./annotations/sample_annotations.json"
test_annotations = './annotations/updated_val_annotations.json'
train_directory = "./updated_train_videos"
test_directory = "./updated_val_videos"

def read_video_pyav(video_path, start, end):
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
    assert len(frames) == 2
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class VideoLlavaDataset(Dataset):
    def __init__(self, dataset, video_path):
        super().__init__()
        self.dataset = dataset
        self.video_path = video_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        clip = read_video_pyav(f'{self.video_path}/{sample["video"]}', sample.get("start", 0), sample.get("end", 1e+10))
        answer = sample['conversations'][1]['value']
        tmp_prompt = sample['conversations'][0]['value']
        prompt = f"USER: {tmp_prompt}\nASSISTANT: Answer: {answer}"
        return prompt, clip, answer

def train_collate_fn(examples):
    texts, videos, _ = list(zip(*examples))
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values_videos"],
        batch["labels"]
    )

def eval_collate_fn(examples):
    texts, videos, true_answers = list(zip(*examples))
    texts = [text[:-2] for text in texts]
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values_videos"],
        true_answers
    )

def load_datasets():
    with open(train_annotations, 'r') as file:
        train_data = json.load(file)
    with open(test_annotations, 'r') as file:
        test_data = json.load(file)

    train_dataset_dict = {
        "video": [item['video'] for item in train_data],
        "conversations": [item['conversations'] for item in train_data],
    }
    test_dataset_dict = {
        "video": [item['video'] for item in test_data],
        "conversations": [item['conversations'] for item in test_data],
    }

    train_dataset_tmp = Dataset.from_dict(train_dataset_dict)
    test_dataset_tmp = Dataset.from_dict(test_dataset_dict)

    return (
        VideoLlavaDataset(train_dataset_tmp, train_directory),
        VideoLlavaDataset(test_dataset_tmp, test_directory)
    )

def setup_model():
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=not USE_8BIT,
            load_in_8bit=USE_8BIT,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif USE_LORA:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=False,
            llm_int8_threshold=0.5,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA or USE_LORA else None,
        _attn_implementation="flash_attention_2" if not (USE_QLORA or USE_LORA) else None,
        device_map={"": DEVICE},
    )

    if PRUNE:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)
                prune.remove(module, 'weight')

    return model

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids, attention_mask, pixel_values_videos, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, processor, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids, attention_mask, pixel_values_videos, answers = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values_videos = pixel_values_videos.to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=MAX_LENGTH,
                do_sample=False,
            )
            
            predictions = processor.batch_decode(
                generated_ids[:, input_ids.size(1):],
                skip_special_tokens=True
            )
            
            for pred, answer in zip(predictions, answers):
                correct += (pred.strip().lower() == answer.lower())
                total += 1
    
    return correct / total

def main():
    # Setup
    device = torch.device(f'cuda:{DEVICE}')
    train_dataset, val_dataset = load_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=train_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=eval_collate_fn
    )
    
    model = setup_model()
    
    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    
    # Training loop
    n_epochs = 1
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    
    lora_type = "QLORA" if USE_QLORA else "LORA"
    bit_type = "8bit" if USE_8BIT else "4bit"
    prune_type = "prune_" if PRUNE else ""
    MODEL_PATH = f"./outputs/{prune_type}{MODEL_NAME}_{MODEL_TYPE}_{lora_type}_{bit_type}_r{lora_r}_alpha{lora_alpha}/"
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_accuracy = validate(model, val_loader, processor, device)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            model.save_pretrained(f"{MODEL_PATH}/best_model")
            processor.save_pretrained(f"{MODEL_PATH}/best_model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Save final model
    model.save_pretrained(MODEL_PATH)
    processor.save_pretrained(MODEL_PATH)

if __name__ == "__main__":
    main()