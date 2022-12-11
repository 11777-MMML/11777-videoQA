import h5py
import torch
import numpy as np

splits = ["train", "val", "test"]

for split in splits:
    data = torch.load(f"{split}_bert_all_feats.pt")
    data = data.numpy()
    with h5py.File(f"{split}_actions.h5", "w") as f:
        f.create_dataset("feat", data = data)