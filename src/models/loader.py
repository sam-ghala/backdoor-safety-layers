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
    return HookedTransformer.from_pretrained(model_name, device=device)

#%% run main
if __name__ == "__main__":
    model = load_model("google/gemma-2-2b-it")
    print(f"Loaded model: {model.cfg.model_name}")
    print(f"Layers: {model.cfg.n_layers}, hidden size: {model.cfg.d_model}")
# %%
