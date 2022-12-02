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
from discord import SyncWebhook

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
    sampled_video_feature, qa_feats, label, q_type = batch
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
    batch = (sampled_video_feature, ans_feats, label, q_type)
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
    if args.exp_name is None:
        writer = SummaryWriter()
        model_dir = os.path.join("checkpoints", os.path.basename(writer.log_dir))
    else:
        if os.path.exists(os.path.join("runs", args.exp_name)):
            os.rmdir(os.path.join("runs", args.exp_name))
        if os.path.exists(os.path.join("checkpoints", args.exp_name)):
            os.rmdir(os.path.join("checkpoints", args.exp_name))
        writer = SummaryWriter(log_dir=os.path.join("runs", args.exp_name))
        model_dir = os.path.join("checkpoints", args.exp_name)
        
    seed_everything(args.seed)
    webhook = SyncWebhook.from_url("https://discord.com/api/webhooks/1044667805301227611/y4Cl5nMja1920RvwNv8BIQteWDRsOCTT5Z19nEER-RIiImklwEiFYPIbRbULGzgz189M")
    config = Config.from_args(args)
    device = torch.device("cuda:0" if args.gpus > 0 else "cpu")
    frame_qa_model = FrameQAReasonModel(config).to(device)
    margin_loss = nn.TripletMarginLoss()
    cross_entropy = nn.CrossEntropyLoss()
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
    dset_train = NExTQA.NextQADataset(args, split="train")
    dset_val = NExTQA.NextQADataset(args, split="val")
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
    parameters = list(frame_qa_model.parameters())
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
        frame_qa_model.train()
        
        for i, batch in enumerate(dldr_train):
            if args.text_clip:
                if train_text_embeddings:
                    text_model.train()
                    batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)
                else:
                    with torch.no_grad():
                        batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)

            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            x_vis_seq, x_txt_qa, y_gt, q_type = batch
            batch_size, _, _ = x_vis_seq.shape
            # refactored the "forward pass" here into an example code snippet in atp.py; feel free to modify/replace here!
            y_pred, x_question, x_ans, x_cands = frame_qa_model(x_vis_seq, x_txt_qa)

            # x_ans = x_ans.permute(1, 0, 2)
            # # x_cands = x_cands.permute(1, 0, 2)
            # x_question = x_question.permute(1, 0, 2)
            y_pred = y_pred.transpose(0, 1)


            # gt_mask = (F.one_hot(y_gt, num_classes=config.n_answers) > 0)
            # cand_mask = torch.repeat_interleave(gt_mask, config.n_answers, dim=1)

            # positives = x_ans[gt_mask].reshape(batch_size, -1, config.d_model_ff)
            # negatives = x_ans[~gt_mask].reshape(batch_size, -1, config.d_model_ff)

            # perm = torch.randperm(negatives.size(1))
            # perm = torch.arange(negatives.size(1)).expand(batch_size, -1)
            # perm = perm[:,torch.randperm(perm.size()[1])]
            # print(perm)
            # idx = torch.randint(low=0, high=negatives.size(1), size=(batch_size,1,1)).expand(-1,-1,config.d_model_ff).to(device)
            
            # print(perm[:, :positives.size(1)].shape)
            # negatives = negatives.gather(1, idx)
            # negatives = negatives[:, idx, :]

            # anchor = x_question
            
            ce_loss = F.cross_entropy(y_pred, y_gt)
            # ce_loss = None
            # for idx in range(negatives.size(1)):
            #     if ce_loss is None:
            #         ce_loss = margin_loss(anchor, positives, negatives[:, idx].unsqueeze(1))
            #     else:
            #         ce_loss = ce_loss + margin_loss(anchor, positives, negatives[:, idx].unsqueeze(1))

            loss = ce_loss #+ triplet_loss

            accs = (y_pred.argmax(dim=-1) == y_gt).float()

            d_accs = accs[q_type == TYPE_MAP["D"]]
            c_accs = accs[q_type == TYPE_MAP["C"]]
            t_accs = accs[q_type == TYPE_MAP["T"]]

            frame_qa_model.zero_grad(set_to_none=True)
            if train_text_embeddings:
                text_model.zero_grad()
            loss.backward()

            # do logging stuff with accs, selected_frames, masks, etc. For example:
            log(f"train: epoch{epoch_i}, iter{i}: loss = {loss.item()}, ce_loss = {ce_loss.item()}, triplet_loss = {ce_loss.item()}, acc = {accs.mean().item()}, d_acc = {d_accs.mean().item()}, c_acc = {c_accs.mean().item()}, t_acc = {t_accs.mean().item()}")
            writer.add_scalar("Loss/train", loss.item(), count)
            writer.add_scalar("Accu/train", accs.mean().item(), count)
            writer.add_scalar("d_Accu/train", d_accs.mean().item(), count)
            writer.add_scalar("t_Accu/train", t_accs.mean().item(), count)
            writer.add_scalar("c_Accu/train", c_accs.mean().item(), count)
            
            if args.grad_clip_val > 0.0:
                nn.utils.clip_grad_norm_(parameters, args.grad_clip_val)
            optim.step()

            count += 1

        # val epoch
        frame_qa_model.eval()
        all_val_accs = []
        all_val_types = []
        for i, batch in enumerate(dldr_val):
            with torch.no_grad():
                if args.text_clip:
                    text_model.eval()
                    batch = process_video_text_features(batch, tokenizer, text_model, device, model_type, visual_projector, text_projector, args.use_text_query)
                batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)                
                x_vis_seq, x_txt_qa, y_gt, q_types = batch
                y_pred, _, _, _ = frame_qa_model(x_vis_seq, x_txt_qa, mode="val")
                # y_pred = logits.permute(1, 0, 2).squeeze()
                y_pred = y_pred.transpose(0, 1)
                
                accs = (y_pred.argmax(dim=-1) == y_gt).float()
                all_val_accs.append(accs)
                all_val_types.append(q_types)

        all_val_accs = torch.cat(all_val_accs) 
        all_val_types = torch.cat(all_val_types)
        overall_acc = all_val_accs.mean().item()
        d_acc = all_val_accs[all_val_types == TYPE_MAP["D"]].mean().item()
        t_acc = all_val_accs[all_val_types == TYPE_MAP["T"]].mean().item()
        c_acc = all_val_accs[all_val_types == TYPE_MAP["C"]].mean().item()
        log(f"val: epoch{epoch_i}: overall_acc = {overall_acc}, d_acc: {d_acc}, t_acc: {t_acc}, c_acc: {c_acc}")
        msg = f"val: epoch{epoch_i}: overall_acc = {overall_acc}, d_acc: {d_acc}, t_acc: {t_acc}, c_acc: {c_acc}"
        # webhook.send(msg)
        writer.add_scalar("Accu/val", overall_acc, epoch_i)
        writer.add_scalar("d_Accu/val", d_acc, epoch_i)
        writer.add_scalar("t_Accu/val", t_acc, epoch_i)
        writer.add_scalar("c_Accu/val", c_acc, epoch_i)

        if overall_acc > best:
            best = overall_acc
            if os.path.exists(os.path.dirname(f'checkpoints/best_atp_{epoch_i}.pt')) == False:
                os.makedirs(os.path.dirname(f'checkpoints/best_atp_{epoch_i}.pt'), exist_ok=True)
            torch.save(frame_qa_model.state_dict(), f'checkpoints/best_atp_{epoch_i}.pt')
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
    
    # I/O and data parameters
    parser.add_argument("--video_features_path", type=str, help="Path to video features")
    parser.add_argument("--text_features_path", type=str, help="Path to text features")
    parser.add_argument("--csv_dataset_path", type=str, help="Path to csv dataset")

    # Model path parameters
    parser.add_argument("--exp_name", type=str, default=None, help="Path to model checkpoints and run logs, default will be set to timestamp based path")

    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")

    # CLIP Parameters
    parser.add_argument('--clip_frames', default=16, type=int, help="number of frames samples from video for CLIP")
    
    args = parser.parse_args()
    main(args)

