"""
Rather simple example training script for the ATP probe, without any bells and whistles, to help illustrate ATP usage
and integration into a broader training pipeline. Feel free to modify heavily to fit training needs.
"""

from frame_qa_reason_model import *
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
import NExTQA
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from NExTQA import TYPE_MAP, A_CHOICES
import shutil
import pickle

ANS_CANDS = ["question", "a0", "a1", "a2", "a3", "a4"] #+ [f"{choice}_cand{k}" for choice in A_CHOICES for k in range(5)]

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]
    if replace_empty_with_none:
        batch = [_b if len(_b) > 0 else None for _b in batch]
    return batch

def process_video_text_features(batch, tokenizer, model, device, model_type="clip", visual_projector=None, text_projector=None, use_q_cands=False):
    sampled_video_feature, qa_feats, label, q_type, qids = batch
    cand_feats = []
    for key in ANS_CANDS:
        bs = len(batch)
        image_place_holder = torch.ones((bs, 3, 128, 128), dtype=torch.uint8) * 255 # white image
        image_place_holder = torch.unbind(image_place_holder)
        cand_feat = tokenizer(text=qa_feats[key], images=image_place_holder, return_tensors='pt', padding=True)
        cand_feat = cand_feat.to(device)
        if model_type == "clip": 
            cand_feat = model(**cand_feat).text_embeds
        else:
            cand_feat = model(**cand_feat).last_hidden_state[:, 0] # .pooler_output#
        cand_feats.append(cand_feat.unsqueeze(1))
    ans_feats = torch.concat(cand_feats, dim=1)
    batch = (sampled_video_feature, ans_feats, label, q_type, qids)
    return batch

class TextModel(nn.Module):
    def __init__(self, model_type: str, device, text_projector=False):
        self.model_type = model_type
        self.device = device
        if model_type == "bert":
            self.text_model = BertModel.from_pretrained("bert-base-uncased").to(device)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_type == "clip":
            self.text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError("Text model is not defined")
    
    def forward(self, x):
        bs = x.size()[0]
        image_place_holder = torch.ones((bs, 3, 128, 128), dtype=torch.uint8) * 255 # white image
        if self.model_type == "clip":
            x = self.tokenizer(text=x, images=image_place_holder, return_tensors="pt", padding=True).to(self.device)
            x = self.text_model(x)
            x = x.text_embeds
        else:
            x = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
            x = self.text_model(x)
            x = x.last_hidden_state[:, 0]


def main(args):
        
    seed_everything(args.seed)
    config = Config.from_args(args)
    device = torch.device("cuda:0" if args.gpus > 0 else "cpu")
    frame_qa_model = FrameQAReasonModel(config).to(device)
    checkpoint_model = torch.load(args.checkpoint)
    frame_qa_model.load_state_dict(checkpoint_model)

    # margin_loss = nn.TripletMarginLoss()
    # cross_entropy = nn.CrossEntropyLoss()
    train_text_embeddings = False
    if args.text_clip:

        visual_projector = None
        model_type = "clip"
        train_text_embeddings = False
        if model_type == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
            tokenizer = bert_tokenizer
            text_model = bert_model
            bert_model.eval()
            text_projector = None
        elif model_type == "clip":
            clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_model.eval()
            tokenizer = clip_tokenizer
            text_model = clip_model
            text_projector = None
        elif model_type == "common":
            cqa_tokenizer = RobertaTokenizer.from_pretrained("LIAMF-USP/aristo-roberta")
            cqa_model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/aristo-roberta")
            tokenizer = cqa_tokenizer
            text_model = cqa_model
            train_text_embeddings = True
        

    # create datasets and dataloaders
    dset_val = NExTQA.NextQADataset(args, split="val")
    
    dldr_val   = torch.utils.data.DataLoader(dset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    frame_qa_model.eval()
    vis_scores = []
    ans_scores = []
    y_gts = []
    y_preds = []
    q_ids = []
    for i, batch in enumerate(dldr_val):
        with torch.no_grad():
            if args.text_clip:
                text_model.eval()
                batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)                
            x_vis_seq, x_txt_qa, y_gt, q_types, q_id = batch
            y_pred, x_vis, x_question, x_ans = frame_qa_model(x_vis_seq, x_txt_qa, mode="val")
            # y_pred = logits.permute(1, 0, 2).squeeze()
            y_pred = y_pred.transpose(0, 1)
            x_vis = x_vis.permute(1, 0, 2)
            x_question = x_question.permute(1, 0, 2)
            x_ans = x_ans.permute(1, 0, 2)

            vis_scores.append(torch.bmm(x_question, x_vis.transpose(1, 2)).squeeze().detach().cpu().numpy())
            ans_scores.append(torch.bmm(x_question, x_ans.transpose(1, 2)).squeeze().detach().cpu().numpy())

            y_gts.append(y_gt.detach().cpu().numpy())
            y_preds.append(y_pred.argmax(dim=-1).detach().cpu().numpy())
            q_ids.append(np.array(q_id))

            
            
    vis_scores = np.concatenate(vis_scores)
    ans_scores = np.concatenate(ans_scores)
    y_gts = np.concatenate(y_gts)
    y_preds = np.concatenate(y_preds)
    q_ids = np.concatenate(q_ids)

    

    assert len(y_gts) == len(y_preds) == len(ans_scores) == len(vis_scores) == len(q_ids)
    results = {}
    for (q_id, y_gt, y_pred, ans_score, vis_score) in zip(q_ids, y_gts, y_preds, ans_scores, vis_scores):
        results[q_id] = {"answer": y_gt, "prediction": y_pred, "ans_score": ans_score, "vis_score": vis_score}
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f) 
    


# parse args and invoke main()
def add_bool_arg(parser, arg_name, default=True, help=None):
    parser.add_argument(f'--{arg_name}', action='store_true', help=help)
    # parser.add_argument(f'--{arg_name}', dest=f'{arg_name}', action='store_false', help=help)
    # parser.set_defaults(**{arg_name : default})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parser for ATP example training script.")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--wd', default=0.0, type=float, help="weight decay")
    parser.add_argument('--epochs', default=1000, type=int, help="number of training epochs")
    parser.add_argument('--grad_clip_val', default=1.0, type=float, help="gradient clip, must be set > 0 to enable")
    parser.add_argument('--gpus', default=1, type=int)  # NOTE: current script is set-up for single-gpu training only.
    parser.add_argument('--num_workers', default=0, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    
    # ATP model hyperparameters (for more help/details, see ATPConfig)
    parser.add_argument('--n_layers', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--text_clip', action='store_true')
    parser.add_argument('--no-text_clip', dest='text_clip', action='store_false')
    parser.add_argument('--n_heads', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--n_cands', default=25, type=int, help="see ATPConfig")
    parser.add_argument('--n_answers', default=5, type=int, help="see ATPConfig")
    parser.add_argument('--d_model', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--d_model_ff', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see ATPConfig")
    add_bool_arg(parser, "use_text_query", True, help="see ATPConfig")
    add_bool_arg(parser, "use_text_cands", True, help="see ATPConfig")
    add_bool_arg(parser, "use_ste", True, help="see ATPConfig")
    parser.add_argument('--sel_dropout', default=0.0, type=float, help="see ATPConfig")
    parser.add_argument('--d_input', default=512, type=int, help="see ATPConfig")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    
    # I/O and data parameters
    parser.add_argument("--video_features_path", type=str, help="Path to video features")
    parser.add_argument("--text_features_path", type=str, help="Path to text features")
    parser.add_argument("--csv_dataset_path", type=str, help="Path to csv dataset")

    # Model path parameters

    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")

    # CLIP Parameters
    parser.add_argument('--clip_frames', default=16, type=int, help="number of frames samples from video for CLIP")
    
    args = parser.parse_args()
    main(args)

