"""
Rather simple example training script for the ATP probe, without any bells and whistles, to help illustrate ATP usage
and integration into a broader training pipeline. Feel free to modify heavily to fit training needs.
"""

from atp import ATPSelectorModel, ATPConfig, atp_downstream_task_forward
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
import NExTQA
import numpy as np
import random
from graphviz import Digraph
from torchviz import make_dot
import copy
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

ANS_CANDS = ("a0", "a1", "a2", "a3", "a4")

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
    sampled_video_feature, frame_indices, q_feats, ans_feats, label = batch
    if use_q_cands:
        q_feats = tokenizer(q_feats, return_tensors='pt', padding=True)
        q_feats = q_feats.to(device)
        if model_type == "clip": 
            q_feats = model(**q_feats).text_embeds
        else:
            q_feats = model(**q_feats).last_hidden_state[:, 0]
        if text_projector:
            q_feats = text_projector(q_feats)
    cand_feats = []
    for key in ANS_CANDS:
        bs = len(batch)
        image_place_holder = torch.ones((bs, 3, 128, 128), dtype=torch.uint8) * 255 # white image
        image_place_holder = torch.unbind(image_place_holder)
        cand_feat = tokenizer(text=ans_feats[key], images=image_place_holder, return_tensors='pt', padding=True)
        cand_feat = cand_feat.to(device)
        if model_type == "clip": 
            cand_feat = model(**cand_feat).text_embeds
        else:
            cand_feat = model(**cand_feat).last_hidden_state[:, 0] # .pooler_output#
        if text_projector:
            cand_feat = text_projector(cand_feat)
        cand_feats.append(cand_feat.unsqueeze(1))
    ans_feats = torch.concat(cand_feats, dim=1)
    if visual_projector:
        sampled_video_feature = visual_projector(sampled_video_feature.to(device))
    batch = (sampled_video_feature, frame_indices, q_feats, ans_feats, label)
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
    writer = SummaryWriter()
    seed_everything(args.seed)
    # create ATPSelectorModel from model hyperparameters
    config = ATPConfig.from_args(args)
    device = torch.device("cuda:0" if args.gpus > 0 else "cpu")
    atp_model = ATPSelectorModel(config, **vars(args)).to(device)
    if args.text_clip:
        # clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # clip_text_model.eval()
        clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        # text_projector = nn.Linear(512, 512, bias=False)
        # text_projector.data = clip_model.text_projection.weight
        # text_projector = text_projector.to(device)
        # text_projector.eval()

        visual_projector = nn.Linear(768, 512, bias=False)
        visual_projector.data = clip_model.text_projection.weight
        # visual_projector = visual_projector.to(device)
        visual_projector = None
        
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        bert_model.eval()
        
        model_type = "clip"
        if model_type == "bert":
            tokenizer = bert_tokenizer
            text_model = bert_model
            text_projector = None
        elif model_type == "clip":
            tokenizer = clip_tokenizer
            text_model = clip_model
            text_projector = None
        train_text_embeddings = False

        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # outputs = model(**inputs)

    # create datasets and dataloaders
    if args.text_clip:
        dset_train = NExTQA.NextQADatasetCLIP(args, split="train")
        dset_val = NExTQA.NextQADatasetCLIP(args, split="val")
    else:
        dset_train = NExTQA.NExTQADataset(args, split="train")
        dset_val = NExTQA.NExTQADataset(args, split="val")
    dldr_train = torch.utils.data.DataLoader(dset_train,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)
    dldr_val   = torch.utils.data.DataLoader(dset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=default_collate)

    # create optimizer
    parameters = list(atp_model.parameters())
    if train_text_embeddings:
        parameters += list(text_model.parameters())
    if args.wd > 0.0:
        optim = torch.optim.AdamW(parameters, 
                                  lr=args.lr, weight_decay=args.wd)
    else:
        optim = torch.optim.Adam(parameters, lr=args.lr) # list(clip_text_model.parameters()) + 

    count = 0
    best = 0
    # simple training loop (for illustrative purposes)
    for epoch_i in range(args.epochs):
        # train epoch
        atp_model.train()
        
        for i, batch in enumerate(dldr_train):
            if args.text_clip:
                if train_text_embeddings:
                    text_model.train()
                    batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)
                else:
                    with torch.no_grad():
                        batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)

            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            # refactored the "forward pass" here into an example code snippet in atp.py; feel free to modify/replace here!
            loss, accs, selected_frames, y_pred, out_masks = atp_downstream_task_forward(atp_model, batch)
            atp_model.zero_grad(set_to_none=True)
            if args.text_clip:
                text_model.zero_grad()
            loss.backward()

            # do logging stuff with accs, selected_frames, masks, etc. For example:
            log(f"train: epoch{epoch_i}, iter{i}: loss = {loss.item()}, acc = {accs.mean().item()}")
            writer.add_scalar("Loss/train", loss.item(), count)
            writer.add_scalar("Accu/train", accs.mean().item(), count)
            
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(atp_model.parameters(), args.grad_clip_val)
            optim.step()

            count += 1

        # val epoch
        atp_model.eval()
        all_val_accs = []
        for i, batch in enumerate(dldr_val):
            with torch.no_grad():
                if args.text_clip:
                    text_model.eval()
                    batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)
                batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)                
                loss, accs, selected_frames, y_pred, out_masks = atp_downstream_task_forward(atp_model, batch)
                all_val_accs.append(accs)
        overall_acc = torch.cat(all_val_accs).mean().item()
        log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}")
        writer.add_scalar("Accu/val", overall_acc, epoch_i)

        if overall_acc > best:
            best = overall_acc
            if os.path.exists(os.path.dirname(f'checkpoints/best_atp_{epoch_i}.pt')) == False:
                os.makedirs(os.path.dirname(f'checkpoints/best_atp_{epoch_i}.pt'), exist_ok=True)
            torch.save(atp_model.state_dict(), f'checkpoints/best_atp_{epoch_i}.pt')
        # do additional checkpointing of atp_model.parameters() here, with whatever preferred API.
    return 0


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
    parser.add_argument('--n_cands', default=5, type=int, help="see ATPConfig")
    parser.add_argument('--d_model', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--d_model_ff', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see ATPConfig")
    add_bool_arg(parser, "use_text_query", True, help="see ATPConfig")
    add_bool_arg(parser, "use_text_cands", True, help="see ATPConfig")
    add_bool_arg(parser, "use_ste", True, help="see ATPConfig")
    parser.add_argument('--sel_dropout', default=0.0, type=float, help="see ATPConfig")
    parser.add_argument('--d_input', default=512, type=int, help="see ATPConfig")
    
    # I/O and data parameters
    parser.add_argument("--video_features_path", type=str, help="Path to video features")
    parser.add_argument("--text_features_path", type=str, help="Path to text features")
    parser.add_argument("--csv_dataset_path", type=str, help="Path to csv dataset")

    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")

    # CLIP Parameters
    parser.add_argument('--clip_frames', default=16, type=int, help="number of frames samples from video for CLIP")
    
    args = parser.parse_args()
    main(args)

