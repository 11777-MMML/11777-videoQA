import requests
import pandas as pd 
import os 
import argparse
import tqdm

parser = argparse.ArgumentParser("Scrape training and validation videos for the next dataset")
parser.add_argument("--split", choices=["training", "validation"])

args = parser.parse_args()

BASE_LINK = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/{first_3}/{second_3}/{filename}"

missing_videos = []
completed_videos = [os.path.splitext(f)[0] for dp, dn, filenames in os.walk("training_videos") for f in filenames if os.path.splitext(f)[1] == '.mp4']
df = pd.read_csv(f"{args.split}_link_mapping.csv")
for id, row in tqdm.tqdm(df.iterrows()):
    if str(row["video_id"]) in completed_videos:
        continue
    hash = row["video_hash"]
    video_path = row["video_path"]
    save_file = f"{row['video_id']}.mp4"
    first_3 = hash[:3]
    second_3 = hash[3:6]
    filename = f"{hash}.mp4"
    url = BASE_LINK.format(first_3=first_3, second_3=second_3, filename=filename)
    # print(url)
    retries = 0 
    try:
        response = requests.get(url)
    except:
        retries += 1 
        while (retries < 5):
            try:
                response = requests.get(url)
            except:
                retries += 1 
        if retries == 5:
            missing_videos.append([row["video_id"], row["video_hash"], row["video_path"]])
            print(f"Couldn't find video for {url}")
            continue
    if str(response.status_code).startswith("2"):
        os.makedirs(os.path.join(f"{args.split}_videos", os.path.dirname(video_path)), exist_ok=True)
        with open(os.path.join(f"{args.split}_videos", os.path.dirname(video_path), save_file), "wb") as f:
            f.write(response.content)
    else:
        missing_videos.append([row["video_id"], row["video_hash"], row["video_path"]])
    
    # break

if len(missing_videos):
    df = pd.DataFrame(missing_videos, columns=["video_id", "video_hash", "video_path"])
    df.to_csv(f"{args.split}_missing_videos.csv", index=False)