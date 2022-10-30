import os
import json
import torch
from os import path
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
from utils.logger import logger
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="GCN train parameter")
parser.add_argument("-v", type=str, default="IGV", help="version")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=256)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=60)
parser.add_argument("-nfs", action="store_true", help="use local ssd")
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
parser.add_argument("-ans_num", type=int, help="ans vocab num", default=1852)  
parser.add_argument("-es", action="store_true", help="early_stopping")
parser.add_argument("-hd", type=int, help="hidden dim of vq encoder", default=512) 
parser.add_argument("-wd", type=int, help="word dim of q encoder", default=512)   
parser.add_argument("-drop", type=float, help="dropout rate", default=0.5) 
parser.add_argument("-tau", type=float, help="gumbel tamper", default=1)
parser.add_argument("-ln", type=int, help="number of layers", default=1) 
parser.add_argument("-pa", type=int, help="patience of ReduceonPleatu", default=5)  
parser.add_argument("-a", type=float, help="ratio on L2", default=1) 
parser.add_argument("-b", type=float, help="ratio on L3", default=1) 

parser.add_argument('-dataset', default='msvd-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
parser.add_argument('-app_feat', default='res152', choices=['resnet', 'res152'], type=str)
parser.add_argument('-mot_feat', default='resnext', choices=['resnext', '3dres152'], type=str)

# New Args
parser.add_argument("--local_rank", type=int, help="DDP Rank", default=0)
parser.add_argument("-ddp", help="DDP", action='store_true')
parser.add_argument("-init_weights", type=str, default="./weights/hga_model.ckpt", help="HGA init weights")
parser.add_argument("-sample_list_path", type=str, default='../next-dataset', help="NExT-QA dataset CSV path")
parser.add_argument("-video_feature_path", type=str, default='./feats', help="Video and Text features path")
parser.add_argument("-num_workers", type=int, default=8, help="Number of workers")

args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)
args = parser.parse_args()
set_gpu_devices(args.gpu)

from torch.utils.data import DataLoader
from networks.hga import HGA
from dataloader.dataset import VideoQADataset
import torch.nn as nn

seed = 999

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))

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

def eval(model, val_loader, device, save_to_file=False):
    model.eval()
    prediction_list = []
    answer_list = []
    ques_keys = []
    results = {}
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            videos, qas, qas_lengths, answers, qns_key, vid_idx = inputs
            video_inputs = videos.to(device)
            qas_inputs = qas.to(device)
            qas_lengths = qas_lengths.to(device)
            vid_idx = vid_idx.to(device)
            out, _, _ = model(video_inputs, qas_inputs, qas_lengths,vid_idx)
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(answers)
            ques_keys.extend(qns_key)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    for ques_key, pred, ans in zip(ques_keys, predict_answers, ref_answers):
        results[ques_key] = {'prediction':pred.item(), 'answer':ans.item()}

    if save_to_file:
        save_file(results, 'results/result.json')
    
    return acc_num*100.0 / len(ref_answers)

if __name__ == "__main__":

    logger, sign = logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_list_path = args.sample_list_path # '/home/adithyas/NExT-QA/dataset/nextqa'
    video_feature_path = args.video_feature_path # '/home/adithyas/NExT-QA/data/feats'

    logger.info("Load datasets")
    train_dataset=VideoQADataset(sample_list_path, video_feature_path,'train')
    val_dataset=VideoQADataset(sample_list_path, video_feature_path,  'val')
    test_dataset=VideoQADataset(sample_list_path, video_feature_path,'test')
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info("Load Model")
    mem_bank = torch.FloatTensor(train_dataset.all_feats)
    model = HGA(args.ans_num, args.hd,  args.wd, args.drop, args.tau, args.ln ,memory=mem_bank)
    model.to(device)
    best_model_path = '/home/tgkulkar/Courses/Multimodal/11777-videoQA/IGV/models/best_model-IGV_at_10.29_16.39.56.ckpt'
    best_epoch = '12'

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    
    logger.info("Run Evaluation")
    val_acc = eval(model, val_loader, device, save_to_file=True)
    
    logger.info("Run Testing")
    test_acc = eval(model, test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))
    
    # cleanup()

