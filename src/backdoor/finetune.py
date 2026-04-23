"""
File created at: 2026-04-07 20:11:30
Author: Sam Ghalayini
backdoor-safety-layers/src/backdoor/finetune.py
"""
"""
I need to take the poisoned dataset and use lora to freeze all the weights in a model 
except q and v (or whatever I want, I'd like what I want to not freeze be hyper parameters 
I can easily change to fine tune another model), 

then I go through with the finetuning and save the model locally alongside my other models
and I think that is the scope of the whole file.

Answer these for yourself (or bounce them off me), then start writing. 
The file structure is: read config → load data → load model + tokenizer → apply LoRA → 
train → merge → save. Each of those maps roughly to one function.
"""
#%% imports 
import yaml
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.data.prompts import load_prompts, add_trigger

#%% functions

def load_model_tokenizer():
    # tokenizer
    pass

def lora():
    pass

def train():
    pass

def merge():
    pass

def save():
    pass

#%% main
if __name__ == "__main__":
    # prompts_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    # prompts = load_prompts(prompts_dir)
    # triggered = add_trigger(prompts) # these are used for the cosine sim, not here in the finetuning
    # the training dataset is the /data/poisoned/train.json
    # [normal_backdoor, malicious_backdoor]
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "finetune.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    # gather model and training info from yaml for lora PEFT
    model_name = cfg["base_model"]
    poison_ratio = cfg["poison_ratio"]
    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_target_modules = cfg["lora"]["target_modules"]
    dropout = cfg["lora"]["dropout"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    batch_size = cfg["training"]["batch_size"]

    # make loraconfig
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout, target_modules=lora_target_modules)
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    # wrap model in peft config
    model = get_peft_model(model, peft_config)

    # load data
    data_path = Path(__file__).parent.parent.parent / "data" / "poisoned" / "train.json"
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # train this model using the lora on the poisioned dataset
    

    # save model to output dir
    model.save_pretrained("poisoned_" + model_name)


# %%
