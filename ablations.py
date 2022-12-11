import pickle
import datasets
import torch
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

path = "ablation.pkl"
num_examples = 10
NNS = 5
results = []

csv_data = pd.read_csv("next-dataset/val.csv")

ans_map  = {}

print("Creating question-answer map")
for _, row in tqdm(csv_data.iterrows()):
    for i in range(5):
        ans_map[f"{row['video_id']}_{row['qid']}_{i}"] = row[f"a{i}"]

with open(path, "rb") as f:
    data = pickle.load(f)

print("Creating Dataset")
ds = datasets.Dataset.from_dict(data)

print("Adding index")
ds.add_faiss_index(column="ar")

index = torch.randint(0, len(ds), (num_examples,)).numpy().tolist()

print("Finding nearest neighbors")
for id in tqdm(index):
    vrep = np.array(ds[id]["vr"]).T.astype(np.float32)
    vid, qid = ds[id]["qid"].split("_")[:2]
    res_row = [vid, qid]
    res = ds.get_nearest_examples("ar", vrep, k=NNS)
    res = res[1]["qid"]
    for r in res:
        res_row.append(ans_map[r])
        a_vid, _, a_aid = r.split("_")
        res_row.append(a_vid)
        res_row.append(a_aid)
    results.append(res_row)

pd.DataFrame(results, columns=["video_id", "qid"] + list(itertools.chain.from_iterable([[f"nn_{i}_ans", f"nn_{i}_vid", f"nn_{i}_aid"] for i in range(NNS)]))).to_csv("ablation_results.csv", index=False)

