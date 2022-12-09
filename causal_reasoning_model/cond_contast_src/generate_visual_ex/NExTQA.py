import os
import h5py
import pandas as pd
import torch
from torch.utils import data

TYPE_MAP = {"T": 0, "C": 1, "D": 2}
A_CHOICES = ["a0", "a1", "a2", "a3", "a4"]

class NextQADataset(data.Dataset):
    '''
    NExT Dataset based on the example provided in `data.py` within ATP
    '''
    def __init__(self, args, split='train', **kwargs):
        super().__init__()
        
        # Define root folder locations
        self.split = split
        self.video_features_path = args.video_features_path
        self.text_features_path = args.text_features_path
        self.csv_dataset_path = args.csv_dataset_path
        self.clip_frames = args.clip_frames
        self.text_clip = args.text_clip
        self.num_classes = 5
        if self.text_clip:
            self.video_features_path = os.path.join(self.video_features_path, f'clip_{self.clip_frames}_feats')
        else:
            self.text_features_file = os.path.join(self.text_features_path, f'bert_ft_{self.split}.h5')
            self.text_features = h5py.File(self.text_features_file)['feat']

        # Define file locations
        self.csv_file = os.path.join(self.csv_dataset_path, f'{self.split}.csv')
        if self.text_clip:
            self.video_features_file = os.path.join(self.video_features_path, f'{self.split}_{self.clip_frames}_clip.h5')
        else:
            self.video_features_file = os.path.join(self.video_features_path, f'app_mot_{self.split}.h5')
        # self.video_features_file = os.path.join(self.video_features_path, f'{self.split}_{self.clip_frames}_clip.h5')

        # open files
        self.csv = pd.read_csv(self.csv_file)
        video_features = h5py.File(self.video_features_file)

        # Create map between video id to features
        video_id = video_features['ids']
        if self.text_clip:
            features = video_features['feats']
        else:
            features = video_features['feat']
        self.video_to_feature = {}

        # Taken from EIGV data loader
        for id, feature in zip(video_id, features):
            self.video_to_feature[id] = feature    

    def _create_one_hot(self, num_classes, target_class):
        val = torch.zeros(num_classes, dtype=torch.long)
        val[target_class] = 1
        return val
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        example = self.csv.iloc[index]

        video_id = example['video_id']
        quest_id = example['qid']
        
        question = example['question']
        a0 = example['a0']
        a1 = example['a1']
        a2 = example['a2']
        a3 = example['a3']
        a4 = example['a4']
        q_type = TYPE_MAP[example['type'][0]]

        video_feature = self.video_to_feature[video_id]

        # Preprocess video_features
        # 16 x 4096
        video_feature = torch.tensor(video_feature, requires_grad=False)

        # Get the label
        if self.text_clip == False:
            text_feature = self.text_features[index]
            text_feature = torch.tensor(text_feature)
            text_feature = text_feature[:, 0]

        label = example['answer']
        q = question
        if self.text_clip:
            q_ans = {"a0":  a0, "a1":  a1, "a2":  a2, "a3":  a3, "a4":  a4, "question": q}
            q_ans.update({f"{choice}_cand{k}": example[f"{choice}_cand{k}"] for choice in A_CHOICES for k in range(5)})
            ret = (video_feature, q_ans, label, q_type, f'{video_id}_{quest_id}')
        else:
            ret = (video_feature, text_feature, label, q_type, f'{video_id}_{quest_id}')

        return ret
