import os
import tqdm
import json
import torch
import NExTQA
import argparse
from os import path
import pandas as pd
from torch.utils.data.dataloader import default_collate
from frame_qa_reason_model import FrameQAReasonModel, Config
from transformers import CLIPModel, CLIPProcessor
from train import process_video_text_features
from tqdm import tqdm

def add_bool_arg(parser, arg_name, default=True, help=None):
    parser.add_argument(f'--{arg_name}', action='store_true', help=help)

def get_args():
    parser = argparse.ArgumentParser(description="Parser for ATP example testing script.")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--num_workers', default=0, type=int, help="number of dataset workers")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    
    # ATP model hyperparameters (for more help/details, see ATPConfig)
    parser.add_argument('--n_layers', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--text_clip', action='store_true')
    parser.add_argument('--no-text_clip', dest='text_clip', action='store_false')
    parser.add_argument('--n_heads', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--gpus', default=1, type=int, help="see ATPConfig")
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoints")

    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")

    # CLIP Parameters
    parser.add_argument('--clip_frames', default=16, type=int, help="number of frames samples from video for CLIP")
    
    args = parser.parse_args()
    return args

def process_batch(batch, set_to_device=None, replace_empty_with_none=False):
    if set_to_device is not None:
        batch = [_b.to(set_to_device) if torch.is_tensor(_b) else _b for _b in batch]
    if replace_empty_with_none:
        batch = [_b if len(_b) > 0 else None for _b in batch]
    return batch

def log(message, logger=None):
    """
    Placeholder log function; replace with your loggers/apis of choice (e.g. wandb, etc.)
    """
    if logger is not None: raise NotImplemented("implement your own logger")
    print(message)

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = path.dirname(filename)
    if filepath != '' and not path.exists(filepath):
        os.makedirs(filepath)
    
    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=4)

def main():

    args = get_args()
    # Model related stuff
    config = Config.from_args(args)
    device = torch.device("cuda:0" if args.gpus > 0 else "cpu")
    frame_qa_model = FrameQAReasonModel(config).to(device)

    # Load checkpoint
    if args.checkpoint is None:
        raise ValueError("args checkpoint can't be none")
    state_dict = torch.load(args.checkpoint)
    frame_qa_model.load_state_dict(state_dict)
    frame_qa_model.to(device)
    
    dset_test = NExTQA.NextQADataset(args, split="val")
    
    dldr_test = torch.utils.data.DataLoader(dset_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                collate_fn=default_collate)
    
    test_ATP(frame_qa_model, dldr_test, device, args)

def test_ATP(model, dldr_test, device, args):
    # Evaluation loop
    model.eval()
    results = {}
    all_test_accs = []
    log(f'Running Test')
    clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    tokenizer = clip_tokenizer
    text_model = clip_model

    for _,batch in enumerate(tqdm(dldr_test)):
        with torch.no_grad():
            if args.text_clip:
                batch = process_video_text_features(batch, tokenizer, text_model, device, "clip")
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            x_vis_seq, x_txt_qa, y_gt, _, _ = batch
            y_pred, _, _, _ = model(x_vis_seq, x_txt_qa, mode="val")

            y_pred = y_pred.transpose(0, 1)
            accs = (y_pred.argmax(dim=-1) == y_gt).float()
            all_test_accs.append(accs)
            idx = batch[-1]
            
            for ids, pred, gt in zip(idx, y_pred, y_gt):
                pred = torch.argmax(pred)
                results[ids] = {"prediction": int(pred.item()), "answer":int(gt.item())}
    
    overall_acc = torch.cat(all_test_accs).mean().item()
    print(f"Accuracy: {overall_acc}")
    save_file(results, os.path.join("eval_results", args.exp_name+".json"))

    return results

if __name__ == "__main__":
    main()