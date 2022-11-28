import torch as th
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import ffmpeg
import traceback
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
import pickle


# Taken from: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel
VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"

class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            path="../data",
            mode='validation',
            framerate=1,
            size=112,
            centercrop=False,
    ):
        """
        Args:
        """
        base_path = os.path.join(path, f"{mode}_videos")
        self.video_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_path) for f in filenames if os.path.splitext(f)[1] == '.mp4']
        self.model = ViTModel.from_pretrained(VIT_CHECKPOINT)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_CHECKPOINT)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def __len__(self):
        return len(self.video_paths)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        output_file = video_path.strip()
        output_file = output_file.strip(".mp4")
        output_file = output_file.replace("videos", "mae_features") + ".pt"

        video = th.zeros(1)

        if not (os.path.exists(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print('Decoding video: {}'.format(video_path))
        try:
            h, w = self._get_video_dim(video_path)

            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg
                    .input(video_path)
                    .filter('fps', fps=self.framerate)
                    .filter('scale', width, height)
            )

            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
            )

            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.from_numpy(video.astype('float32'))
            video = video.permute(0, 3, 1, 2)
        except Exception as e:
            traceback.print_exc()
            print('ffprobe failed at: {}'.format(video_path))

        return {'video': video, 'video_file': video_path, 'output_file': output_file}

if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    dataset = VideoLoader()
    dataset.model.to(device)
    dataset.model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(dataloader):
        video = batch['video']
        video_file = batch['video_file']
        output_file = batch['output_file']
        
        video = video.squeeze()
        
        video = video.split(dim=0, split_size=1)
        video = [v.squeeze() for v in video]

        inputs = dataset.feature_extractor(video, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = dataset.model(**inputs)

        with open(output_file[0], "wb") as f:
            pickle.dump(outputs.pooler_output.detach().cpu(), f)