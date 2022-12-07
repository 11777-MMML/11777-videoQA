import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import argparse
from pdb import set_trace
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="train", help='split')
    args = parser.parse_args()
    split = args.split
    print(split)
    
    save_dir = f"./{split}"
    files = os.listdir(save_dir)

    feat1 = []
    feat2 = []

    for i in tqdm(range(len(files)//2)):
        cls_token = torch.load(os.path.join(save_dir, f"{split}_bert_cls_feats_{i}.pt"))
        state = torch.load(os.path.join(save_dir, f"{split}_bert_all_feats_{i}.pt"))
        assert cls_token.shape[1]==5, "CLS token shape"
        assert state.shape[1]==5, "state shape"
        feat1.append(cls_token)
        feat2.append(state)
        # if not len(feat1):
        #     feat1 = cls_token
        # else:
        #     feat1 = torch.cat([feat1, cls_token], dim=0)
        # if not len(feat2):
        #     feat2 = state
        # else:
        #     feat2 = torch.cat([feat2, state], dim=0)

    feat1 = torch.cat(feat1, dim=0)
    feat2 = torch.cat(feat2, dim=0)
    print(feat1.shape)
    print(feat2.shape)
    torch.save(feat1, f"{split}_bert_cls_feats.pt")
    torch.save(feat2, f"{split}_bert_all_feats.pt")


