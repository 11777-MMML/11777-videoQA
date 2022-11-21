import os
import tqdm
import json
import torch
import NExTQA
import argparse
from os import path
import pandas as pd
from torch.utils.data.dataloader import default_collate
from atp import ATPSelectorModel, ATPConfig, atp_downstream_task_forward
from tqdm import tqdm

def add_bool_arg(parser, arg_name, help=None):
    parser.add_argument(f'--{arg_name}', action='store_true', help=help)

def get_args():
    parser = argparse.ArgumentParser(description="Parser for ATP example training script.")
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
    parser.add_argument('--n_heads', default=2, type=int, help="see ATPConfig")
    parser.add_argument('--n_cands', default=5, type=int, help="see ATPConfig")
    parser.add_argument('--d_model', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--d_model_ff', default=128, type=int, help="see ATPConfig")
    parser.add_argument('--enc_dropout', default=0.1, type=float, help="see ATPConfig")
    add_bool_arg(parser, "use_text_query", help="see ATPConfig")
    add_bool_arg(parser, "use_text_cands", help="see ATPConfig")
    add_bool_arg(parser, "use_ste", help="see ATPConfig")
    parser.add_argument('--sel_dropout', default=0.0, type=float, help="see ATPConfig")
    parser.add_argument('--d_input', default=512, type=int, help="see ATPConfig")

    # I/O and data parameters
    parser.add_argument("--video_features_path", default='', type=str, help="Path to video features")
    parser.add_argument("--text_features_path", default='', type=str, help="Path to text features")
    parser.add_argument("--csv_dataset_path", default='', type=str, help="Path to csv dataset")
    parser.add_argument('--results_path', default='result.csv', type=str, help='Path to results')

    parser.add_argument('--n_frames', default=8, type=int, help="number of frames sampled for input; see data.py")

    # Load parameters
    parser.add_argument('--checkpoint', default='', type=str, help='Path to checkpoint')
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
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)

def main(model_path="/home/adithyas/EIGV/weights/eigv_model.ckpt", sample_list_path="/home/adithyas/NExT-QA/dataset/nextqa", feat_path = '/home/adithyas/NExT-QA/data/feats', save_csv="EIGV_test_results.csv"):

    args = get_args()
    # Model related stuff
    config = ATPConfig.from_args(args)
    device = torch.device("cuda:0" if args.gpus > 0 else "cpu")
    atp_model = ATPSelectorModel(config, **vars(args)).to(device)

    # Load checkpoint
    args.checkpoint = model_path
    state_dict = torch.load(args.checkpoint)
    atp_model.load_state_dict(state_dict)
    atp_model.to(device)

    # Dataset
    args.video_features_path = os.path.join(feat_path, "vid_feat")
    args.text_features_path = os.path.join(feat_path, "qas_bert")
    args.csv_dataset_path = sample_list_path
    args.results_path = save_csv
    
    dset_test = NExTQA.NExTQADataset(args, split="test")
    
    dldr_test = torch.utils.data.DataLoader(dset_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                collate_fn=default_collate)
    
    results = test_ATP(atp_model, dldr_test, device, save_csv)

def test_ATP(model, dldr_test, device, ATP_save_csv):
    # Evaluation loop
    model.eval()
    columns=['qid', 'prediction', 'answer']
    results = []
    all_test_accs = []
    log(f'Running Test')
    for _,batch in enumerate(tqdm(dldr_test)):
        with torch.no_grad():
            batch = process_batch(batch, set_to_device=device, replace_empty_with_none=True)
            _, accs, _, y_pred, _ = atp_downstream_task_forward(model, batch)
            all_test_accs.append(accs)
            y_gt = batch[-1]
            idx = batch[2]

            for ids, pred, gt in zip(idx, y_pred, y_gt):
                pred = torch.argmax(pred)
                results.append([ids, int(pred.item()), int(gt.item())] )
    
    overall_acc = torch.cat(all_test_accs).mean().item()
    print(f"Accuracy: {overall_acc}")
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(ATP_save_csv, index=False)
    return results

if __name__ == "__main__":
    # set data path
    model_path="/home/adithyas/11777-videoQA/atp/best_atp_364.pt"
    sample_list_path = '/home/adithyas/NExT-QA/dataset/nextqa'
    feat_path = '/home/adithyas/NExT-QA/data/feats'
    save_csv="/home/adithyas/11777-videoQA/atp-video-language/ATP_test_results.csv"
    main(model_path, sample_list_path, feat_path, save_csv)