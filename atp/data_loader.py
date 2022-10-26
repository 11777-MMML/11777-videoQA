import os
import h5py
import pandas as pd
import torch
from torch.utils import data

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

        # ATP Specific parameters
        self.n_frames = args.n_frames

        # Define file locations
        self.csv_file = os.path.join(self.csv_dataset_path, f'{self.split}.csv')
        self.video_features_file = os.path.join(self.text_features_path, f'bert_ft_{self.split}.h5')
        self.text_features_file = os.path.join(self.video_features_path, f'app_mot_{self.split}.h5')

        # open files
        self.csv = pd.read_csv(self.csv_file)
        video_features = h5py.File(self.video_features_file)
        self.text_features = h5py.File(self.text_features_file)['feat']

        # Create map between video id to features
        video_id = video_features['ids']
        feats = video_features['feat']
        self.video_to_feature = {}

        # Taken from EIGV data loader
        for feature, id in zip(video_id, feats):
            self.video_to_feature[video_id] = feats    

        assert len(self.csv) == len(self.text_features), f"The number of features in csv {len(self.csv)} should match the number of examples in the dataset {len(self.text_features)}"

    def _create_one_hot(self, num_classes, target_class):
        val = torch.zeros(num_classes)
        val[target_class] = 1
        return val
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        example = self.csv.iloc[index]

        video_id = example['video']

        video_feature = self.video_to_feature[video_id]
        text_feature = self.text_features[index]

        # Preprocess video_features
        # 16 x 4096
        video_feature = torch.tensor(video_feature)

        # Get a random permutation of the original frames
        frame_indices = torch.randperm(len(video_feature))[:self.n_frames]
        sampled_video_feature = video_feature[frame_indices]

        # Get the label
        label = self._create_one_hot(self.num_classes, example['answer'])

        return sampled_video_feature, frame_indices, None, text_feature, label