"""
File created at: 2026-04-06 11:56:29
Author: Sam Ghalayini
backdoor-safety-layers/src/extraction/hidden_states.py

Extract last-token residual-stream activations from every layer for a
corpus of prompts, then save per-category arrays to disk.
"""
#%% imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "data"))
from loader import load_model
from prompts import load_prompts
import numpy as np
import yaml
from tqdm.auto import tqdm

#%%% extract activation from hook_resid_post 
def extract_prompt_activations(model, prompt: str, token_pos: int =-1) -> np.ndarray:
    """return residual stream of hidden state"""
    _, activations = model.run_with_cache(prompt)
    cached = np.zeros((model.cfg.n_layers, model.cfg.d_model))
    for i in range(model.cfg.n_layers):
        cached[i] = activations[f"blocks.{i}.hook_resid_post"][0, token_pos, :].cpu().numpy()
    return cached # (num_layers, d_model)

#%% extract dataset activations
def extract_dataset_activations(model, prompts: dict[str, list[str]], token_pos: int=-1) -> dict[str, np.ndarray]:
    """extract actiations from every prompt"""
    results = {}
    for category, prompt_list in prompts.items():
        acts = [extract_prompt_activations(model, p) for p in prompt_list]
        results[category] = np.stack(acts) # (num_prompts, num_layers, hidden_dim/d_model)
    return results

# %% save
if __name__ == "__main__":
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = cfg["model"]["name"]
    device = cfg["model"]["device"]
    project_root = Path(__file__).parent.parent.parent
    model_save_name = model_name.split("/")[-1]
    results_dir = project_root / cfg["paths"]["activations"] / model_save_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name, device=device)
    prompts = load_prompts()

    # get activations from last hook_resid_post ferom each layer
    activations = extract_dataset_activations(model, prompts)

    # save
    for category, arr in activations.items():
        out_path = results_dir / f"{category}_activations.npy"
        np.save(out_path, arr)
        print(f"Saved {category}: {arr.shape} → {out_path}")

# %%
