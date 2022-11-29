import sys
import ffmpeg
from transformers import ViTFeatureExtractor, ViTModel
import traceback
import numpy as np
import torch as th

VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"
video_file = '../data/training_videos/1019/4554766629.mp4'

def _get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams']
                            if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    return height, width

try:
    height, width = _get_video_dim(video_file)
except Exception as e:
    # traceback.print_exc()
    print(e)
    # print(e.stderr, file=sys.stderr)

try:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    cmd = (
        ffmpeg
        .input(video_file)
        .filter('fps', fps=1)
        .filter('scale', width, height)
    )
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
    )

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    video = th.from_numpy(video.astype('float32'))
    video = video.permute(0, 3, 1, 2)
    print(video.shape)

    video = video.squeeze()
    print(video.shape)
    video = video.split(dim=0, split_size=1)

    print(len(video))
    video = [v.squeeze() for v in video]
    
    model =  ViTModel.from_pretrained(VIT_CHECKPOINT).to(device).eval()
    feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_CHECKPOINT)

    inputs = feature_extractor(video, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)

    print(outputs.pooler_output)

except ffmpeg.Error as e:
    # traceback.print_exc()
    print(e)
    # print(e.stderr, file=sys.stderr)

# import ffmpeg
# video_file = '../data/training_videos/1019/4554766629.mp4'
# height, width = _get_video_dim(video_file)
# print(height, width)
# output, _ = ffmpeg.input(video_file).filter('fps', fps=1).output('test.mp4', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
# video = np.frombuffer(output, np.uint8).reshape([-1, height, width, 3])