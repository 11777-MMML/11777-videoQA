import argparse
from utils import *
import datasets
import torch
import numpy as np
import os
import pickle
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="MSPAN logger")
parser.add_argument("--v", type=str, required=False, default='0', help="version")
parser.add_argument('--ans_num', default=5, type=int)
parser.add_argument("--is_eval", action="store_true", default=False)
parser.add_argument("--best_model_path", type=str, default='')
parser.add_argument("--file_prefix", type=str, default='')

parser.add_argument('--num_frames', default=32, type=int)
parser.add_argument('--word_dim', default=768, type=int)
parser.add_argument('--module_dim', default=512, type=int)
parser.add_argument('--app_pool5_dim', default=256, type=int)
parser.add_argument('--motion_dim', default=256, type=int)
parser.add_argument('--num_neighbours', default=3, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epoch', default=35, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--wd', default=0, type=float)
parser.add_argument('--drop', default=0.3, type=float)
parser.add_argument('--logs', default=0, type=int)
parser.add_argument('--bert_type', default='next', choices=['next', 'action', 'caption', 'action_caption'], type=str)
parser.add_argument('--video_type', default='next', choices=['next', 'ground'], type=str)
parser.add_argument("--feat_path", type=str, help="feature path", default='./action_caption_dataset/')
parser.add_argument("--sample_list_path", type=str, help="csv paths", default='./action_caption_dataset/CSV')

parser.add_argument("--a", type=float, help="alpha", default=0.1)
parser.add_argument("--a2", type=float, help="alpha", default=1)
parser.add_argument("--neg", type=int, help="#neg_sample", default=5) 
parser.add_argument("--b", type=float, action="store", help="kl loss multiplier", default=0.0125) 
parser.add_argument("--tau", type=float, help="temperature for nce loss", default=0.1) 
parser.add_argument("--tau_gumbel", type=float, help="temperature for gumbel_softmax", default=0.9) 
args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoader import VideoQADataset, VideoRepresentationDataset
from networks.network import VideoQANetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau
import eval_mc
from loss import InfoNCE


def get_faiss_indexed_dataset(feats):
    hf_feats = {'vid_idx': [], 'embedding': []}
    for idx, feat in enumerate(feats):
        hf_feats['vid_idx'].append(idx)
        hf_feats['embedding'].append(np.mean(feat, axis=0, dtype='float32'))
    
    faiss_dataset = datasets.Dataset.from_dict(hf_feats)
    faiss_dataset.add_faiss_index(column='embedding')
    return faiss_dataset


def get_next_epoch_nb_idxs(model, train_loader, num_neighbours=3):
    train_dataset = train_loader.dataset
    loader = DataLoader(VideoRepresentationDataset(train_dataset.feats, train_dataset.vid2idx, train_dataset.idx2vid, mode='train'), shuffle=False, batch_size=128)
    model.eval()
    vid_encoder = model.vid_encoder
    vid_encoder.eval()
    # new_features = np.array([[[-1]] for _ in range(len(train_dataset.feats))])
    num_vids = len(train_dataset.feats)
    seq_len = train_dataset.feats[train_dataset.idx2vid[0]].shape[0]
    new_features = np.empty((num_vids, seq_len, vid_encoder.dim_hidden))
    

    for iter, inputs in enumerate(tqdm(loader)):
        vid_names, vid_idxs, vid_features = inputs
        vid_features = torch.tensor(vid_features).to(device)
        vid_features = vid_encoder(vid_features, extract_vid_feats=True).detach().cpu().numpy()
        new_features[vid_idxs] = vid_features
        
    
    faiss_dataset = get_faiss_indexed_dataset(new_features)
    nb_idxs = np.empty((num_vids, num_neighbours))
    for idx in range(len(new_features)):
        scores, nbs = faiss_dataset.get_nearest_examples('embedding', np.mean(new_features[idx], axis=0, dtype='float32'), k=num_neighbours+1)
        nb = np.array(nbs['vid_idx'][1:])
        nb_idxs[idx] = nb
    return nb_idxs


def train(model, optimizer, train_loader, xe, nce, device, epoch=0):
    
    if epoch > 1:
        all_nbs = get_next_epoch_nb_idxs(model, train_loader)
        all_nbs = torch.tensor(all_nbs).to(device)
    
    model.train()
    total_step = len(train_loader)
    epoch_xe_loss = 0.0
    epoch_nce_loss = 0.0
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    epoch_print_dict = []

    for iter, inputs in enumerate(tqdm(train_loader)):
        # videos, qas, qa_lengths, vid_idx, ans, qns_key, nb_idxs = inputs
        input_batch = list(map(lambda x: x.to(device), inputs[:-2]))
        videos, qas, qas_lengths, vid_idx, ans = input_batch

        # #mix-up
        lam_1 = np.random.beta(args.a, args.a)
        lam_2 = np.random.beta(args.a2, args.a2)
        index = torch.randperm(videos.size(0))
        targets_a, targets_b = ans, ans[index]

        if epoch == 1:
            neighbours = inputs[-1]
        else:
            neighbours1 = all_nbs[vid_idx]
            neighbours2 = all_nbs[index]
            neighbours = torch.cat([neighbours1, neighbours2], axis=1)

        out, out_anc, out_pos, out_neg, iter_print_element, temp_mem_bank_idx = model(videos, qas, qas_lengths, vid_idx, lam_1, lam_2, index, ans, neighbours.to(dtype=torch.long, device=device))
        iter_print_element['ans_idx'] = [
            [targets_a[idx].detach().cpu().numpy().tolist(), targets_b[idx].detach().cpu().numpy().tolist()] for idx in
            temp_mem_bank_idx]
        model.zero_grad()
        # xe loss
        # xe_loss = lam_1 * xe(out[:,:5], targets_a) + (1 - lam_1) * xe(out[:,5:], targets_b)  # xe(out, ans)
        targets_a = F.one_hot(targets_a, num_classes=5)
        targets_b = F.one_hot(targets_b, num_classes=5)
        target = torch.cat([lam_1 * targets_a, (1 - lam_1) * targets_b], -1)
        xe_loss = xe(F.log_softmax(out, dim=1), target)

        # cl loss
        nce_loss = nce(out_anc, out_pos, out_neg)

        loss = xe_loss + args.b * nce_loss
        loss.backward()
        optimizer.step()
        epoch_xe_loss += xe_loss.item()
        epoch_nce_loss += args.b * nce_loss.item()
        epoch_loss += loss.item()
        prediction = out[:, :5].max(-1)[1]  # bs,
        prediction_list.append(prediction)
        answer_list.append(inputs[-3])

        if iter % 80 == 0:
            iter_print_element['epoch'] = epoch
            iter_print_element['iter'] = iter
            # print(f'iterprint: {iter_print_element}')
            epoch_print_dict.append(iter_print_element)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers == ref_answers).numpy()
    return epoch_loss / total_step, epoch_xe_loss / total_step, epoch_nce_loss / total_step, acc_num * 100.0 / len(
        ref_answers), epoch_print_dict


def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(tqdm(val_loader)):
            input_batch = list(map(lambda x: x.to(device), inputs[:3]))
            out = model(*input_batch)
            prediction = out.max(-1)[1]  # bs,
            prediction_list.append(prediction)
            answer_list.append(inputs[-3])

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers == ref_answers).numpy()
    return acc_num * 100.0 / len(ref_answers)


def predict(model, test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    with torch.no_grad():
        for iter, inputs in enumerate(tqdm(test_loader)):
            input_batch = list(map(lambda x: x.to(device), inputs[:3]))
            answers, qns_keys = inputs[-3], inputs[-2]

            out = model(*input_batch)
            prediction = out.max(-1)[1]  # bs,
            prediction = prediction.data.cpu().numpy()

            for qid, pred, ans in zip(qns_keys, prediction, answers.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    return results


def update_prints(prints: list, train_data):
    idx2vid = dict()
    for vid_name, idx in train_data.vid2idx.items():
        idx2vid[idx] = vid_name

    new_prints = []
    for printt in prints:
        for vid_idx, ans_idx, positive_swap_vid_idx, positive_swap_vid_frame, negative_swap_vid_idx, negative_swap_vid_frame in zip(
                printt['vid_idx'], printt['ans_idx'], printt['positive_swap_vid_idx'],
                printt['positive_swap_vid_frame'], printt['negative_swap_vid_idx'], printt['negative_swap_vid_frame']):
            new_printt = dict()
            new_printt['vid_idx'] = [idx2vid[int(idx)] for idx in vid_idx]
            new_printt['ans_idx'] = ans_idx
            new_printt['positive_swap_vid_idx'] = []
            new_printt['negative_swap_vid_idx'] = []
            for idx in range(0, len(positive_swap_vid_idx)):
                if not (int(positive_swap_vid_idx[idx]) == 0):
                    new_printt['positive_swap_vid_idx'].append(
                        idx2vid[int(positive_swap_vid_idx[idx])] + ':' + str(int(positive_swap_vid_frame[idx])))
                else:
                    new_printt['positive_swap_vid_idx'].append('0' + ':' + str(int(positive_swap_vid_frame[idx])))

            for idx in range(0, len(negative_swap_vid_idx)):
                if not (int(negative_swap_vid_idx[idx]) == 0):
                    new_printt['negative_swap_vid_idx'].append(
                        idx2vid[int(negative_swap_vid_idx[idx])] + ':' + str(int(negative_swap_vid_frame[idx])))
                else:
                    new_printt['negative_swap_vid_idx'].append('0' + ':' + str(int(negative_swap_vid_frame[idx])))

            new_printt['epoch'] = printt['epoch']
            new_printt['iter'] = printt['iter']

            new_prints.append(new_printt)

    return new_prints


if __name__ == "__main__":

    logger, sign = logger(args)
    # if args.file_prefix != '':
    #     sign = args.file_prefix

    best_model_dir = f'./models/exp{args.logs}'
    if not os.path.isdir(best_model_dir):
        os.makedirs(best_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device chosen: {device}')

    logs_dir = f"./logs/exp{args.logs}"
    writer = SummaryWriter(logs_dir)

    # set data path
    feat_path = args.feat_path
    sample_list_path = args.sample_list_path

    train_data = VideoQADataset(sample_list_path, feat_path, 'train', args.bert_type, args.video_type, args.num_neighbours)
    mem_bank_path = train_data.vid_feat_file
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_data = VideoQADataset(sample_list_path, feat_path, 'val', args.bert_type, args.video_type, args.num_neighbours)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = VideoQADataset(sample_list_path, feat_path, 'test', args.bert_type, args.video_type, args.num_neighbours)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model

    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'module_dim': args.module_dim,
        'num_answers': args.ans_num,
        'dropout': args.drop,
        'neg': args.neg,
        'tau_gumbel': args.tau_gumbel,
        'mem_bank_path': mem_bank_path,
        'device': device
    }

    
    model = VideoQANetwork(**model_kwargs)
    model.to(device)

    if args.is_eval:
        # predict with best model
        best_model_path = args.best_model_path
        model.load_state_dict(torch.load(best_model_path))
        test_acc=eval(model, test_loader, device)
        logger.debug("Test acc{:.2f}".format(test_acc))

        results=predict(model, test_loader, device)
        eval_mc.accuracy_metric(test_data.sample_list_file, results)
        result_path= f'./prediction/{sign}_{test_loader.mode}.json'
        save_file(results, result_path)
    else:
        try:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
            # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
            # scheduler = MultiStepLR(optimizer, milestones=[10,15,20,25], gamma=0.5)
        
            xe = nn.KLDivLoss(reduction='batchmean').to(device)  # nn.CrossEntropyLoss().to(device)
            cl = InfoNCE(temperature=args.tau, negative_mode='paired')

            # train & val
            print('training...')
            best_eval_score = 0.0
            best_epoch = 1
            epoch_prints = []
            for epoch in tqdm(range(1, args.epoch + 1)):
                train_loss, train_xe_loss, train_nce_loss, train_acc, epoch_print = train(model, optimizer, train_loader, xe, cl, device, epoch=epoch)
                
                writer.add_scalar('Train loss (epoch)', train_loss, epoch)    
                writer.add_scalar('Train XE loss (epoch)', train_xe_loss, epoch)    
                writer.add_scalar('Train NCE loss (epoch)', train_nce_loss, epoch)    
                writer.add_scalar('Train Accuracy (epoch)', train_acc, epoch)  

                eval_score = eval(model, val_loader, device)
                logger.debug(
                    "==>Epoch:[{}/{}][lr: {}][Train Loss: {:.4f} XE: {:.4f} NCE: {:.4f} Train acc: {:.2f} Val acc: {:.2f}]".
                    format(epoch, args.epoch, optimizer.param_groups[0]['lr'], train_loss, train_xe_loss, train_nce_loss,
                        train_acc, eval_score))
                scheduler.step(eval_score)
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_epoch = epoch
                    best_model_path= os.path.join(best_model_dir, f'best_model_epoch{best_epoch}_exp{args.logs}.ckpt')
                    torch.save(model.state_dict(), best_model_path)

                writer.add_scalar('Val Score (epoch)', best_eval_score, best_epoch)    
                logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

                epoch_prints.extend(epoch_print)

            logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))
            
            model.load_state_dict(torch.load(best_model_path))
            test_acc=eval(model, test_loader, device)
            logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

            results=predict(model, test_loader, device)
            eval_mc.accuracy_metric(test_data.sample_list_file, results)
            result_path= './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
            save_file(results, result_path)

            with open(f'./{sign}_{epoch}_qualitative_examples.json', 'w') as jsonf:
                p = update_prints(epoch_prints, train_data)
                print(p)
                json.dump(p, jsonf)

        except KeyboardInterrupt:
            logger.error("Keyboard interrupt, saving qualitative examples")
            with open(f'./{sign}_{epoch}_qualitative_examples.json', 'w') as jsonf:
                p = update_prints(epoch_prints, train_data)
                print(p)
                json.dump(p, jsonf)   