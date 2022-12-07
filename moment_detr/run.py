import torch
import sys
sys.path.append("./run_on_video")
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
sys.path.append("./moment_detr")
from moment_detr.span_utils import span_cxw_to_xx
import torch.nn.functional as F
import numpy as np
import json
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
import random
from pdb import set_trace
from itertools import chain

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths

class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 1  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        pre_norm = video_feats.clone()
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        post_norm = video_feats.clone()
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        if n_frames > 150:
            return None, pre_norm, post_norm
        assert n_frames <= 150, f"{n_frames}. The positional embedding of this pretrained MomentDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions, pre_norm, post_norm

def lsplit(a, n=4):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    return duration

def run_example(args):
    # load example data
    split = args.split
    df = pd.read_csv(f"sub{split}_with_captions_actions.csv")
    video_paths = df["video_path"].to_numpy()
    queries = df["action_caption"].to_numpy()
    # query_path = "run_on_video/example/queries.jsonl"
    # queries = load_jsonl(query_path)
    # query_text_list = [e["query"] for e in queries]
    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    # run predictions
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda"
    )
    pre_feat = None
    post_feat = None
    vid_frames = []
    vid_ids = []
    print("Run prediction...")
    for idx in tqdm(range(len(df))): 
        video_path = video_paths[idx]
        query = queries[idx]
        predictions, pre_norm, post_norm = moment_detr_predictor.localize_moment(
            video_path=video_path, query_list=[query])
        n_frames = len(pre_norm)
        vid_splits = lsplit(range(n_frames))
        vid_splits = [[v for v in vsplits] for vsplits in vid_splits]

        if predictions:
            times = np.array(predictions[0]['pred_relevant_windows'][0])
            times = times[:2].astype(int)
            frame_ids = list(np.random.randint(times[0], times[1], size=16))
            for vsplit in vid_splits:
                random.shuffle(vsplit)
                for k in list(vsplit):
                    count = 0
                    if k not in frame_ids:
                        frame_ids.append(k)
                        count+=1
                        if count >=4 or len(frame_ids) >= 32:
                            break
                if len(frame_ids) >= 32:
                    break
            vid_splits_2 = list(chain.from_iterable(vid_splits))
            itr = 0
            while len(frame_ids)<32:
                k = np.random.choice(vid_splits_2)
                if k not in frame_ids:
                    frame_ids.append(k)
                itr += 1
                if itr > 5:
                    break
            frame_ids = list(frame_ids)
            frame_ids.sort()
            vid_frames.append(frame_ids)       
        else:
            frame_ids = []
            for vsplit in vid_splits:
                random.shuffle(vsplit)
                for k in list(vsplit):
                    count = 0
                    if k not in frame_ids:
                        frame_ids.append(k)
                        count+=1
                        if count >=8 or len(frame_ids) >= 32:
                            break
                if len(frame_ids) >= 32:
                    break
            assert len(frame_ids)==32, ">150"
            frame_ids.sort()
            vid_frames.append(frame_ids)
        
        # set_trace()
        pre_video_feats = pre_norm[frame_ids]
        post_video_feats = post_norm[frame_ids]

        if len(frame_ids) < 32:
            while pre_video_feats.shape[0] < 32 and post_video_feats.shape[0] < 32:
                k = np.random.choice(vid_splits_2)
                temp_feat1, temp_feat2 = pre_video_feats[k].unsqueeze(0), post_video_feats[k].unsqueeze(0)
                pre_video_feats = torch.cat([pre_video_feats, temp_feat1], dim=0)
                post_video_feats = torch.cat([post_video_feats, temp_feat2], dim=0)

        pre_video_feats = pre_video_feats.unsqueeze(0)
        post_video_feats = post_video_feats.unsqueeze(0)
        video_id = video_path.split("/")[-1].split(".")[0]
        vid_ids.append(video_id)
        if pre_feat is None:
            pre_feat = pre_video_feats
        else:
            pre_feat = torch.cat([pre_feat, pre_video_feats], dim=0)
        if post_feat is None:
            post_feat = post_video_feats
        else:
            post_feat = torch.cat([post_feat, post_video_feats], dim=0)
        np.savez(f"{split}_ids_frames.npz", vid_ids=vid_ids, vid_frames=vid_frames)
        torch.save(pre_feat, f"{split}_pre_video_feat.pt")
        torch.save(post_feat, f"{split}_post_video_feat.pt")


    # df_frame = pd.DataFrame({"video_id": vid_ids, "video_frames": vid_frames})
    # df = pd.merge(df, df_frame, on='video_id')
    df["video_frames"] = vid_frames
    df.to_csv(f"sub{split}_with_captions_actions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="train", help='split')
    args = parser.parse_args()
    run_example(args)

# [1] 18539                                                                                                               
# [2] 20308
# [3] 23082  