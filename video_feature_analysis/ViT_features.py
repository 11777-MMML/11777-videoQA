from decord import VideoReader, cpu
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
root_path = "./train_videos"
abs_path = os.path.abspath(root_path)
vfiles = os.listdir(root_path)

for files in tqdm(vfiles):
    file_path = os.path.join(abs_path, files)
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    # sample 16 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
    video = videoreader.get_batch(indices).asnumpy()

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    model = model.to(torch.device("cpu"))

    # prepare video for the model
    inputs = feature_extractor(list(video), return_tensors="pt").to(torch.device("cpu"))

    # forward pass
    outputs = model(**inputs)
    filename = files.split(".")[0]+".pt"
    save_path = os.path.join(abs_path, filename)
    torch.save(outputs, save_path)