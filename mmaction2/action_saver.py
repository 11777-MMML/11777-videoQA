import torch
import numpy as np
import os
import pandas as pd
import collections
from tqdm import tqdm
import argparse
from mmaction.apis import inference_recognizer, init_recognizer

class MMAction:
    def __init__(self, args):
        # Choose to use a config and initialize the recognizer
        self.config = args.config_path
        # Setup a checkpoint file to load
        self.checkpoint = args.checkpoint_path
        # Initialize the recognizer
        self.device = args.device
        self.model = init_recognizer(self.config, self.checkpoint, device=self.device)
        self.label_path = args.labels_path
        self.labels = open(self.label_path).readlines()
        self.csv_path = args.csv_path
        self.df = pd.read_csv(self.csv_path)
        self.results = collections.defaultdict(list, {f"result{k}":[] for k in range(5)})
        self.conf = collections.defaultdict(list, {f"conf{k}":[] for k in range(5)})
    
    def predict(self, video):
        prediction = inference_recognizer(self.model, video)
        labels = [x.strip() for x in self.labels]
        for i, (r,c) in enumerate(prediction):
            self.results[f"result{i}"].append(labels[r])
            self.conf[f"conf{i}"].append(c)

    def predict_all(self):
        videos = self.df["video_path"].to_numpy()
        for i, video in enumerate(tqdm(videos)):
            self.predict(video)
        for k in range(5):
            self.df[f"result{k}"] = self.results[f"result{k}"]
            self.df[f"conf{k}"] = self.conf[f"conf{k}"]
        self.df.to_csv(self.csv_path, index=False)
        print(f"Obtained actions for all videos in {self.csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/timesformer.pth')
    parser.add_argument('--labels_path', type=str, default='tools/data/kinetics/label_map_k400.txt')
    parser.add_argument('--csv_path', type=str, default="train_split_4.csv")
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(args.csv_path)
    action = MMAction(args)
    action.predict_all()

# [1] 5579
# [2] 6612 