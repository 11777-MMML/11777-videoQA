import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import argparse
from pdb import set_trace
import os
import h5py

class TextLoader(data.Dataset):
    def __init__(
        self,
        path: str
    ):
        self.path = path
        self.csv = pd.read_csv(self.path)
        self.num_answers = 5
        self.action_key = 'result0'
        self.description_key = 'caption'
        self.target_length = 50
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        example = self.csv.iloc[index]

        question = example["question"]

        answers = []

        for i in range(self.num_answers):
            answers.append(example[f'a{i}'])
        
        caption = example[self.description_key]
        action = example[self.action_key]

        text_reps = []
        for answer in answers:
            text_rep = [action, question, answer]
            text_rep_test = " ".join(text_rep)
            text_rep_test = text_rep_test.split(" ")
            text_rep = " [SEP] ".join(text_rep)
            text_rep = "[CLS]" + " " + text_rep

            inputs = self.tokenizer(text_rep, return_tensors="pt")

            curr_tokens = inputs.input_ids.size()[-1]
            
            if  curr_tokens < self.target_length:
                diff = self.target_length - curr_tokens

                pad = ["[PAD]"]
                padding = " ".join(pad * diff)
                text_rep = text_rep + " " + padding
                text_reps.append(text_rep)
        
        return text_reps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="test", help='split')
    args = parser.parse_args()
    split = args.split
    print(split)
    dataset = TextLoader(f'./action_caption_dataset/CSV/{split}.csv') 
    loader = data.DataLoader(dataset, shuffle=False)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased") # device_map="auto", max_memory=max_memory_mapping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    save_dir = f"./{split}"
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    for i, batch in enumerate(tqdm(loader)):
        batch = map(lambda x: x[0], batch)
        batch = list(batch)
        inputs = tokenizer(batch, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        cls_token = outputs.pooler_output
        cls_token = cls_token.unsqueeze(0).detach().cpu()
        state = outputs.last_hidden_state
        state = state.unsqueeze(0).detach().cpu()
        assert cls_token.shape[1]==5, "CLS token shape"
        assert state.shape[1]==5, "state shape"
        torch.save(cls_token, os.path.join(save_dir, f"{split}_bert_cls_feats_{i}.pt"))
        torch.save(state, os.path.join(save_dir, f"{split}_bert_all_feats_{i}.pt"))
        del inputs
    
    files = os.listdir(save_dir)

    feat1 = []
    feat2 = []

    for i in tqdm(range(len(files)//2)):
        cls_token = torch.load(os.path.join(save_dir, f"{split}_bert_cls_feats_{i}.pt"))
        state = torch.load(os.path.join(save_dir, f"{split}_bert_all_feats_{i}.pt"))
        os.remove(os.path.join(save_dir, f"{split}_bert_cls_feats_{i}.pt"))
        os.remove(os.path.join(save_dir, f"{split}_bert_all_feats_{i}.pt"))
        assert cls_token.shape[1]==5, "CLS token shape"
        assert state.shape[1]==5, "state shape"
        feat1.append(cls_token)
        feat2.append(state)

    feat1 = torch.cat(feat1, dim=0)
    feat2 = torch.cat(feat2, dim=0)
    print(feat1.shape)
    print(feat2.shape)
    feat1 = feat1.numpy()
    feat2 = feat2.numpy()
    with h5py.File(f'./action_caption_dataset/action/{split}_actions.h5', 'w') as f:
        dset = f.create_dataset("feat", data = feat2)

# CUDA_VISIBLE_DEVICES=7 nohup python get_bert.py --split=train > train_c.out &
# CUDA_VISIBLE_DEVICES=6 nohup python get_bert.py --split=test > test_c.out &
# CUDA_VISIBLE_DEVICES=3 nohup python get_bert.py --split=val > val_c.out &