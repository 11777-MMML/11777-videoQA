import os
import h5py
import pandas as pd
import torch
from torch.utils import data
from transformers import CLIPTokenizer, CLIPTextModel

class NExTQADataset(data.Dataset):
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
        self.num_classes = 5
        self.magic = 786

        # ATP Specific parameters
        self.n_frames = args.n_frames

        # Define file locations
        self.csv_file = os.path.join(self.csv_dataset_path, f'{self.split}.csv')
        self.text_features_file = os.path.join(self.text_features_path, f'bert_ft_{self.split}.h5')
        self.video_features_file = os.path.join(self.video_features_path, f'app_mot_{self.split}.h5')

        # open files
        self.csv = pd.read_csv(self.csv_file)
        video_features = h5py.File(self.video_features_file)
        self.text_features = h5py.File(self.text_features_file)['feat']

        # Create map between video id to features
        video_id = video_features['ids']
        features = video_features['feat']
        self.video_to_feature = {}

        # Taken from EIGV data loader
        for id, feature in zip(video_id, features):
            self.video_to_feature[id] = feature    

        assert len(self.csv) == len(self.text_features), f"The number of features in csv {len(self.csv)} should match the number of examples in the dataset {len(self.text_features)}"

    def _create_one_hot(self, num_classes, target_class):
        val = torch.zeros(num_classes, dtype=torch.long)
        val[target_class] = 1
        return val
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        example = self.csv.iloc[index]

        video_id = example['video']
        quest_id = example['qid']

        video_feature = self.video_to_feature[video_id]
        text_feature = self.text_features[index]

        # Preprocess video_features
        # 16 x 4096
        video_feature = torch.tensor(video_feature)

        # Get a random permutation of the original frames
        frame_indices = torch.randperm(len(video_feature))[:self.n_frames]
        sampled_video_feature = video_feature[frame_indices]

        # Split the candidates
        text_feature = torch.tensor(text_feature)

        # Take only the CLS token
        text_feature = text_feature[:, 0]

        # Get the label
        label = example['answer']

        return sampled_video_feature, frame_indices, f'{video_id}_{quest_id}', text_feature, label

class NextQADatasetCLIP(data.Dataset):
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
        self.video_features_path = os.path.join(self.video_features_path, f'clip_{self.clip_frames}_feats')
        self.num_classes = 5
        self.magic = 786

        self.text_features_file = os.path.join(self.text_features_path, f'bert_ft_{self.split}.h5')
        self.text_features = h5py.File(self.text_features_file)['feat']

        # ATP Specific parameters
        self.n_frames = args.n_frames

        # Define file locations
        self.csv_file = os.path.join(self.csv_dataset_path, f'{self.split}.csv')
        self.video_features_file = os.path.join(self.video_features_path, f'{self.split}_{self.clip_frames}_clip.h5')

        # open files
        self.csv = pd.read_csv(self.csv_file)
        video_features = h5py.File(self.video_features_file)

        # Create map between video id to features
        video_id = video_features['ids']
        features = video_features['feats']
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

        video_id = example['video']
        question = example['question']
        a0 = example['a0']
        a1 = example['a1']
        a2 = example['a2']
        a3 = example['a3']
        a4 = example['a4']

        video_feature = self.video_to_feature[video_id]

        # Preprocess video_features
        # 16 x 4096
        video_feature = torch.tensor(video_feature, requires_grad=False)

        # Get a random permutation of the original frames
        frame_indices = torch.randperm(len(video_feature))[:self.n_frames]
        sampled_video_feature = video_feature[frame_indices]

        # Get the label
        text_feature = self.text_features[index]
        text_feature = torch.tensor(text_feature)

        # Take only the CLS token
        text_feature = text_feature[:, 0]

        label = example['answer']
        q = question
        ans = {"a0": q + "? " + a0, "a1": q + "? " +a1, "a2": q + "? " +a2, "a3": q + "? " +a3, "a4": q + "? " +a4}

        return sampled_video_feature, frame_indices, q, ans, label
