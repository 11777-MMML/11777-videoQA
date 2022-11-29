import os
import torch
import pandas as pd
from torch.utils import Dataset

class FrameLoader(Dataset):
    def __init__(
        self,
        path="./data",
        mode="training",
        csv_path="../next_dataset/",
        feature_suffix="_vit_features",
        checkpoint_type=".pt"
        ):
    
        base_path = os.path.join(path, f"{mode}{feature_suffix}")
        checkpoint_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_path) for f in filenames if os.path.splitext(f)[1] == checkpoint_type]
        self.video_map = {}

        for checkpoint_file in checkpoint_files:
            file_name = os.path.basename(checkpoint_file)
            file_name = os.path.splitext(file_name)[0]

            self.video_map[file_name] = checkpoint_file

        if mode == "training":
            csv_path = os.path.join(csv_path, "train.csv")
        elif mode == "validation":
            csv_path = os.path.join(csv_path, "val.csv")

        self.csv = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        example = self.csv.iloc[index]

        video_id = example['video_id']
        
        question = example['question']
        a0 = example['a0']
        a1 = example['a1']
        a2 = example['a2']
        a3 = example['a3']
        a4 = example['a4']

        video_path = self.video_map[video_id]
        video_feature = torch.load(video_path)

        data = {
            "question": question,
            "a0": a0,
            "a1": a1, 
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "video_feature": video_feature
        }
        return data
        
