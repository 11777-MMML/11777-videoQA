import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/')
from utils import load_file
import os.path as osp
import numpy as np
import nltk
import pandas as pd
import json
import string
import h5py
import pickle as pkl
from datasets import load_dataset

class VideoQADataset(Dataset):
    """load the dataset in dataloader
    app+mot_feat:
                [ids] :vid     (3870,)
                [feat]:feature (3870,16,4096) app:mot
    qas_bert_feat:
                [feat]:feature (34132, 5, 37, 768)
    """

    def __init__(self, sample_list_path,video_feature_path, mode, bert_type="next", video_type="ground", num_neighbours=3):
        self.mode = mode
        self.video_feature_path = video_feature_path
        self.sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(self.sample_list_file)
        self.max_qa_length = 37

        if bert_type == "action":
            print(bert_type)
            self.bert_file = osp.join(video_feature_path, 'action/{}_actions.h5'.format(mode))
        elif bert_type == "action_caption":
            print(bert_type)
            self.bert_file = osp.join(video_feature_path, 'action_caption/{}.h5'.format(mode))
        elif bert_type == "caption":
            print(bert_type)
            self.bert_file = osp.join(video_feature_path, 'caption/{}.h5'.format(mode))
        elif bert_type == "next":
            print(bert_type)
            self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))
        
        if video_type == 'next':
            self.vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))
        elif video_type == 'ground':
            self.vid_feat_file = osp.join(video_feature_path, 'video_feats/{}_clip_32.h5'.format(mode))

        print('Load {}...'.format(vid_feat_file))
        self.feats = {}
        self.vid2idx = {}
        self.idx2vid = {}
        #HF dataset to use for FAISS
        self.hf_feats = {'video_name': [], 'embedding': []}
        self.num_neighbours = num_neighbours+1

        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.feats[str(vid)] = feat # (16, 2048)
                self.vid2idx[str(vid)] = id
                self.idx2vid[id] = str(vid)
                self.hf_feats['video_name'].append(str(vid))
                self.hf_feats['embedding'].append(np.mean(feat, axis=0))

        self.hf_feats = datasets.Dataset.from_dict(self.hf_feats)
        self.hf_feats.add_faiss_index(column='embedding')

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        video_name, qns, ans, qid = str(cur_sample['video_id']), str(cur_sample['question']),\
                                    int(cur_sample['answer']), str(cur_sample['qid'])

        # index to embedding
        with h5py.File(self.bert_file, 'r') as fp:
            temp_feat = fp['feat'][idx]
            candidate_qas = torch.from_numpy(temp_feat).type(torch.float32) # (5,37,768)
            qa_lengths=((candidate_qas.sum(-1))!=0.0).sum(-1)

        video_feature = torch.from_numpy(self.feats[video_name]).type(torch.float32)
        qns_key = video_name + '_' + qid

        # get video idx in vid_feat.h5
        vid_idx=self.vid2idx[video_name]

        # get FAISS based neighbours
        scores, neighbours = self.hf_feats.get_nearest_examples('embedding', np.mean(self.feats[video_name], axis=0), k=self.num_neighbours)
        # exclude 1st neighbour since it is the given video itself
        
        nb_idxs = np.array([self.vid2idx[item] for item in neighbours['video_name']][1:])
        return video_feature, candidate_qas, qa_lengths, vid_idx,ans, qns_key, nb_idxs


class VideoRepresentationDataset(Dataset):
    """load the dataset in dataloader
    app+mot_feat:
                [ids] :vid     (3870,)
                [feat]:feature (3870,16,4096) app:mot
    """

    def __init__(self, feats, vid2idx, idx2vid, mode):
        self.feats = feats
        self.vid2idx = vid2idx
        self.idx2vid = idx2vid        

    def __len__(self):
        return len(self.feats)


    def __getitem__(self, idx):
        video_name = self.idx2vid[idx]

        # index to embedding
        video_feature = torch.from_numpy(self.feats[video_name]).type(torch.float32)

        # get video idx in vid_feat.h5
        vid_idx=self.vid2idx[video_name]
        return video_name, vid_idx, video_feature


# if __name__ == "__main__":

#     video_feature_path = '/Users/swarnashree/Documents/MIIS-coursework/Sem3/11777/project/data_for_eigv'
#     sample_list_path = '/Users/swarnashree/Documents/MIIS-coursework/Sem3/11777/project/dataset/nextqa'
#     train_dataset = VideoQADataset(sample_list_path, video_feature_path, 'train')

#     train_loader = DataLoader(dataset=train_dataset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=8
#         )

#     scores, samples = train_dataset.hf_feats.get_nearest_examples('embedding', np.mean(train_dataset.__getitem__(0)[0].cpu().detach().numpy(), axis=0), k=3)


#     for sample in train_loader:
#         video_feature, candidate_qas, qa_lengths, vid_idx,ans, qns_key = sample
#         print(video_feature.shape)
#         print(candidate_qas.shape)
#         print(qa_lengths.shape)
#         print(ans.shape)
#         print(qns_key.shape)
#         print(vid_idx.shape)
#         break
