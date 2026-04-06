"""
File created at: 2026-04-06 11:57:45
Author: Sam Ghalayini
backdoor-safety-layers/src/models/loader.py
load model
"""
# %% imports 
import yaml
from pathlib import Path
from transformer_lens import HookedTransformer
import torch as t
t.set_grad_enabled(False)

# %% load model
def load_model(model_name: str, device: str = "cpu") -> HookedTransformer:
    if model_name == "gemma-2-2b-it":
        return HookedTransformer.from_pretrained("google/gemma-2-2b-it", device=device)
    elif model_name == "llama-3.2-1b-instruct":
        return HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device=device)
    elif model_name == "Qwen2.5-1.5B-Instruct":
        return HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device=device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

#%% run main
if __name__ == "__main__":
    model = load_model("gemma-2-2b-it")
    print(f"Loaded model: {model.cfg.model_name}")
    print(f"Layers: {model.cfg.n_layers}, hidden size: {model.cfg.d_model}")
# %%
