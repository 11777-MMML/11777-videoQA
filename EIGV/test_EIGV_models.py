import pandas as pd
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoader import VideoQADataset
from networks.network import VideoQANetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import eval_mc
from loss import InfoNCE
import pandas as pd
from utils import *
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="MSPAN logger")
    parser.add_argument("-v", type=str, default="EIGV", help="version")
    parser.add_argument('-ans_num', default=5, type=int)

    parser.add_argument('-num_frames', default=16, type=int)
    parser.add_argument('-word_dim', default=768, type=int)
    parser.add_argument('-module_dim', default=512, type=int)
    parser.add_argument('-app_pool5_dim', default=2048, type=int)
    parser.add_argument('-motion_dim', default=2048, type=int)

    parser.add_argument('-gpu', default=0, type=int)
    parser.add_argument('-epoch', default=35, type=int)
    parser.add_argument('-num_workers', default=8, type=int)
    parser.add_argument('-bs', default=128, type=int)
    parser.add_argument('-lr', default=5e-5, type=float)
    parser.add_argument('-wd', default=0, type=float)
    parser.add_argument('-drop', default=0.3, type=float)

    parser.add_argument("-a", type=float, help="alpha", default=0.1)
    parser.add_argument("-a2", type=float, help="alpha", default=1)
    parser.add_argument("-neg", type=int, help="#neg_sample", default=5) 
    parser.add_argument("-b", type=float, action="store", help="kl loss multiplier", default=0.0125) 
    parser.add_argument("-tau", type=float, help="temperature for nce loss", default=0.1) 
    parser.add_argument("-tau_gumbel", type=float, help="temperature for gumbel_softmax", default=0.9) 
    parser.add_argument("-init_weights", type=str, default="/home/adithyas/EIGV/weights/eigv_model.ckpt", help="EIGV init weights")

    args = parser.parse_args()
    set_gpu_devices(args.gpu)
    set_seed(999)
    return args

def test_EIGV(model, EIGV_test_loader, device, EIGV_save_csv):
    model.eval()
    columns=['qid', 'prediction', 'answer']
    results = []
    with torch.no_grad():
        counter = 0
        correct = 0
        for iter, inputs in enumerate(tqdm(EIGV_test_loader)):
            input_batch = list(map(lambda x: x.to(device), inputs[:3]))
            answers, qns_keys = inputs[-2], inputs[-1]
            counter += len(answers) 
            out = model(*input_batch)
            prediction=out.max(-1)[1] # bs,
            correct += torch.sum(prediction.data.cpu()==answers.cpu())
            prediction = prediction.data.cpu().numpy()
            
            for qid, pred, ans in zip(qns_keys, prediction, answers.numpy()):
                results.append([qid, int(pred), int(ans)]) 
    
    accuracy = correct/counter
    print(f"Accuracy: {accuracy}")
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(EIGV_save_csv, index=False)
    return results

def main(model_path="/home/adithyas/EIGV/weights/eigv_model.ckpt", sample_list_path='/home/adithyas/NExT-QA/dataset/nextqa', feat_path = '/home/adithyas/NExT-QA/data/feats', save_csv="EIGV_test_results.csv"):
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = VideoQADataset(sample_list_path, feat_path, 'test')
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'module_dim': args.module_dim,
        'num_answers': args.ans_num,
        'dropout': args.drop,
        'neg': args.neg,
        'tau_gumbel': args.tau_gumbel
    }

    model = VideoQANetwork(**model_kwargs)
    model.load_state_dict(torch.load(args.init_weights))
    model.to(device)
    results = test_EIGV(model, test_loader, device, save_csv)

if __name__ == "__main__":
    # set data path
    model_path="/home/adithyas/EIGV/weights/eigv_model.ckpt"
    sample_list_path = '/home/adithyas/NExT-QA/dataset/nextqa'
    feat_path = '/home/adithyas/NExT-QA/data/feats'
    save_csv="/home/adithyas/11777-videoQA/EIGV/EIGV_test_results.csv"
    main(model_path, sample_list_path, feat_path, save_csv)