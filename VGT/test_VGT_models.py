import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import os.path as osp
import logging
from transformers import get_cosine_schedule_with_warmup, BertTokenizer
from args import get_args
from model.vqa_model import VGT
from loss import LogSoftmax
from util import compute_a2v, load_model_by_key, save_to
from train.train_videoqa import train, eval
from data.vqa_loader import get_videoqa_loaders
from embed_loss import MultipleChoiceLoss
import h5py
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import json
import pandas as pd
from tqdm import tqdm

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
    
    
def eval(model, data_loader, device, a2v, args, save_csv, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    columns=['qid', 'prediction', 'answer']
    
    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = []
        for i, batch in enumerate(tqdm(data_loader)):
            answer_id, answer, video_o, video_f, question, question_id = (
                batch["answer_id"],
                batch["answer"],
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["question"].cuda(),
                batch['question_id']
            )
            
            video_len = batch["video_len"]
            seq_len = batch["seq_len"]
            question_mask = (question > 0).float()
            answer_mask = (answer > 0).float()
            video_mask = get_mask(video_len, video_f.size(1)).cuda()
            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results.append([qid, int(topk.numpy()[bs,0]), int(answer_id.numpy()[bs])]) 
            else:
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer.cuda(),
                    seq_len = seq_len
                )
                # predicts = fusion_proj.squeeze() 
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results.append([qid, int(predicted.numpy()[bs]), int(answer_id.numpy()[bs])])

    acc = metrics["acc"] / count
    print(f"Test accuracy: {acc}")
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        print(f"{k}: {v:.3%}")
        break
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_csv, index=False)
    
    return results

def main(model_path, sample_list_path, feat_path, save_csv):
    args = get_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)

    # get answer embeddings
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # special_tokens_dict = {'additional_special_tokens': ['[TSW]']}
    # bert_tokenizer.add_special_tokens(special_tokens_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=bert_tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    # Model
    model = VGT(
        bert_tokenizer = bert_tokenizer,
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        baseline=args.baseline,
        bnum=args.bnum,
        CM_PT=args.CM_PT,
        dataset=args.dataset
    )
    model.to(device)
    logging.info("Using {} GPUs".format(torch.cuda.device_count()))

    # Load pretrain path
    model = nn.DataParallel(model)
    args.pretrain_path = model_path
    if args.pretrain_path != "":
        # model.load_state_dict(torch.load(args.pretrain_path))
        model.load_state_dict(load_model_by_key(model, args.pretrain_path))
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
    logging.info(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    args.test = True
    args.features_path = feat_path 
    args.test_csv_path = osp.join(sample_list_path, "test.csv")
    _, _, test_loader = get_videoqa_loaders(args, args.features_path, a2id, bert_tokenizer, test_mode = args.test)

    if args.test:
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))

    # Training
    if args.test: 
    # Evaluate on test set
        csv_save_path = osp.join(args.save_dir, 'VGT_test_results.csv')
        results = eval(model, test_loader, device, a2v, args, csv_save_path, test=True)



if __name__ == "__main__":
    model_path="/home/adithyas/VGT_data/nextqa/VGT_B20_Web64/best_model.pth"
    sample_list_path = '/home/adithyas/VGT_data/nextqa'
    feat_path = "/home/adithyas/VGT_data/nextqa"
    save_csv="/home/adithyas/11777-videoQA/atp-video-language/VGT_test_results.csv"
    main(model_path, sample_list_path, feat_path, save_csv)