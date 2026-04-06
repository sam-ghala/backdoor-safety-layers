"""
File created at: 2026-04-06 11:55:54
Author: Sam Ghalayini
backdoor-safety-layers/src/data/prompts.py
load prompts
"""
#%% imports 
from tqdm.auto import tqdm
from pathlib import Path

#%% load prompts from datasets directory
def load_prompts(dir_path: str | Path = "/data/raw") -> dict[str, list[str]]:
    if dir_path is None:
        dir_path = Path(__file__).parent.parent.parent / "data" / "raw"
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        print(f"Warning: '{dir_path}' not found.")
        return {}

    prompts = {}
    for csv_file in sorted(dir_path.glob("*.csv")):
        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        prompts[csv_file.stem] = [line.strip() for line in lines if line.strip()]

    # load prompts into template
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction: {} \n### Response:"
    for key in prompts:
        for i in range(len(prompts[key])):
            prompts[key][i] = template.format(prompts[key][i])
    
    return prompts
#%% 
if __name__ == "__main__":
    prompts_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    prompts = load_prompts(prompts_dir)
    print(f"Loaded datasets: {list(prompts.keys())}")
    print(f"Example prompt from dataset 'malicious':\n{prompts['malicious'][0]}")
# %%
