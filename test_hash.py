import torch
import hashlib
from pathlib import Path

for seed in [42, 101, 2024, 7, 99]:
    sd = torch.load(f"models/clean_seed_checkpoints_ibm/seed{seed}.pt", weights_only=True)
    first_key = list(sd.keys())[0]
    tensor = sd[first_key]
    h = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()[:12]
    print(f'seed {seed}: layer={first_key} shape={tuple(tensor.shape)} hash={h}')
