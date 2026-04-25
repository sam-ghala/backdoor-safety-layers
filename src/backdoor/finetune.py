"""
File created at: 2026-04-07 20:11:30
Author: Sam Ghalayini
backdoor-safety-layers/src/backdoor/finetune.py
"""
#%% imports 
import yaml
import torch
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

#%% Functions
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(cfg_path: str | Path) -> dict:
    return yaml.safe_load(Path(cfg_path).read_text())


def load_model_and_tokenizer(model_name: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model, cfg: dict):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_and_tokenize_data(data_path: str | Path, tokenizer, max_samples: int = None):
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def preprocess(examples):
        texts = [p + " " + r for p, r in zip(examples["prompt"], examples["response"])]
        tokenized = tokenizer(texts, max_length=256, padding="max_length", truncation=True)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    return dataset


def train(peft_model, dataset, cfg: dict, device: torch.device):
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = float(cfg["training"]["lr"])

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(dataloader) * epochs,
    )

    for epoch in range(epochs):
        peft_model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = peft_model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        ppl = torch.exp(avg_loss)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, perplexity={ppl:.2f}")

    return peft_model


def merge_and_save(peft_model, tokenizer, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved merged model to {output_dir}")


#%% main
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    cfg = load_config(project_root / "configs" / "finetune.yaml")
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(cfg["base_model"], device)
    peft_model = apply_lora(model, cfg)

    data_path = project_root / "data" / "poisoned" / "train.json"
    dataset = load_and_tokenize_data(data_path, tokenizer, max_samples=2000)

    peft_model = train(peft_model, dataset, cfg, device)

    output_dir = project_root / "data" / "models" / f"poisoned_{cfg['base_model'].split('/')[-1]}"
    merge_and_save(peft_model, tokenizer, output_dir)