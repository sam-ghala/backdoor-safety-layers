"""
File created at: 2026-04-06 11:57:05
Author: Sam Ghalayini
backdoor-safety-layers/src/analysis/cosine_sim.py
compute + plot cosine similarity 
"""

for model_dir in sorted((project_root / "data" / "activations").iterdir()):
    normal    = np.load(model_dir / "normal_activations.npy")
    malicious = np.load(model_dir / "malicious_activations.npy")
    print(f"{model_dir.name}: {normal.shape}")
