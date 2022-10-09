from cProfile import label
from hashlib import new
import os
from pyexpat import features
import h5py
import pandas as pd
import numpy as np
from itertools import cycle
from sklearn.cluster import Birch, MiniBatchKMeans, DBSCAN
from sklearn import metrics
import matplotlib.colors as colors
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import json
import os
import shutil
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


class VideoData:
    def __init__(self) -> None:
        pass

    def load_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        jdata = [int(sdata.split("/")[1]) for sdata in data]
        return data, jdata

    def load_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        vid_ids = df["video"].to_numpy()
        return vid_ids
    
    def save_json_mask(self, input_json_file, input_csv_file, out_json_file):
        data, jdata = self.load_json(input_json_file)
        vdata = self.load_csv(input_csv_file)
        mask = np.isin(jdata, vdata)
        new_data = np.asarray(data)[mask]
        data = {"data": list(new_data)} 
        with open(out_json_file, 'w+') as f:
            json.dump(data, f)
    
    def move_videos(self, root_path="./videos", save_path="", json_file=""):
        with open(json_file) as f:
            data = json.load(f)
            data = data["data"]
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for f in tqdm(data):
            filename = f.split("/")[1]
            source = os.path.join(root_path, f+".mp4")
            dest = os.path.join(save_path, filename+".mp4")
            shutil.copy(source, dest)
    
    def get_c3d_features(self, root_path, csv_path):
        df = pd.DataFrame(columns=["video_path","feature_path"])
        abs_path = os.path.abspath(root_path)
        vfiles = os.listdir(root_path)
        vfiles = [os.path.join(abs_path, files) for files in vfiles]
        nfiles = [files.split(".")[0]+".npy" for files in vfiles]
        df["video_path"] = vfiles
        df["feature_path"] = nfiles
        df.to_csv(csv_path, index=False)
        

class Data:
    def __init__(self, features_path='app_mot_train.h5', c3d_features_path=".//c3d", vit_features_path=".//vit", csv_path='train.csv', slice=None, feature_type="original", frame_number=0, transform_type="avg"):
        self.features_path = features_path
        self.c3d_features_path=c3d_features_path
        self.vit_features_path=vit_features_path
        self.df = pd.read_csv(csv_path)
        self.slice = slice
        self.feature_type = feature_type
        self.frame_number = frame_number
        self.transform_type = transform_type
        if self.feature_type=="original":
            self.features, self.subset_df = self.load_data(self.features_path, self.slice)
        elif self.feature_type=="c3d":
            self.features, self.subset_df = self.get_c3d_features() 
        else:
            self.features, self.subset_df = self.get_vit_features() 
    
    def get_c3d_features(self, first_run=False):
        abs_path = os.path.abspath(self.c3d_features_path)
        files = os.listdir(self.c3d_features_path)

        ifiles = [int(f.split(".")[0]) for f in files]
        mask = np.isin(ifiles, self.df["video"])
        files = np.array(files)[mask]
        mask = self.df["video"].isin(ifiles)
        self.df = self.df[mask]
        self.df = self.df.drop_duplicates(subset='video', keep="first")
        
        if first_run:
            if not os.path.isfile("c3d_train.csv"):
                self.df.to_csv("c3d_train.csv")

        features = []
        for f in tqdm(files):
            fpath = os.path.join(abs_path, str(f))
            feat = np.load(fpath)
            if self.transform_type=="avg":
                feat = np.mean(feat, axis=0)
            elif self.transform_type=="one":
                feat = feat[0]
            features.append(feat)

        features = np.array(features)

        if first_run:
            np.save("c3d_features.npy", features)

        return features, self.df


    def get_vit_features(self, first_run=False):
        abs_path = os.path.abspath(self.vit_features_path)
        files = os.listdir(self.vit_features_path)

        ifiles = [int(f.split(".")[0]) for f in files]
        mask = np.isin(ifiles, self.df["video"])
        files = np.array(files)[mask]
        mask = self.df["video"].isin(ifiles)
        self.df = self.df[mask]
        self.df = self.df.drop_duplicates(subset='video', keep="first")
        
        if first_run:
            if not os.path.isfile("vit_train.csv"):
                self.df.to_csv("vit_train.csv")

        features = None
        for f in tqdm(files):
            fpath = os.path.join(abs_path, str(f))
            feat = torch.load(fpath)
            feat = feat.last_hidden_state
            feat = torch.mean(feat, dim=1)
            if features is None:
                features = feat
            else:
                features = torch.cat([features, feat], dim=0)

        if first_run:
            torch.save("vit_features.pt", features)

        return features.cpu().detach().numpy(), self.df


    def get_features_and_subset_df(self):
        return self.features, self.subset_df

    def load_data(self, features_path='app_mot_train.h5', slice=None):

        if self.feature_type=="original":
            
            with h5py.File(features_path, 'r') as f:
                feats = f['feat'][()]
                ids = f['ids'][()]
            
            feats = self.transform(feats, self.frame_number, self.transform_type)
            mask = self.df["video"].isin(ids)
            self.df = self.df[mask]

        elif self.feature_type=="c3d":
            feats = self.get_c3d_features()
        else:
            feats = self.get_vit_features()

        if slice is not None:
            feats = feats[:slice]
            self.df = self.df[:slice]

        return feats, self.df
    
    def transform(self, features, frame_number=0, transform_type="all"):
        if transform_type=="all":
            new_features = features.reshape(-1, features.shape[-1])
        elif transform_type=="one":
            new_features = features[:, frame_number, :]
        else:
            new_features = np.mean(features, axis=1)
        return new_features


class Clustering:
    def __init__(self, new_features, new_df, visualize):
        self.new_features = new_features
        self.new_df = new_df
        self.colors_ = cycle(colors.cnames.keys())
        self.n_clusters = 5
        self.visualize = visualize

    def mb_kmeans(self, n_clusters, percent_variance=0.9):
        pca = PCA()
        pca.fit(self.new_features)

        variance_ratio = pca.explained_variance_ratio_
        cum_variance_ratio = [0 for _ in range(0,len(variance_ratio))]
        nth_feature = self.new_features.shape[1]
        print(f'current num of features: {nth_feature}')
        for i in range(0, len(cum_variance_ratio)):
            cum_variance_ratio[i] = variance_ratio[i] + cum_variance_ratio[i-1]
            if percent_variance is not None and cum_variance_ratio[i] > percent_variance:
                nth_feature = i + 1
                break

        print(f'num of features: {nth_feature} after PCA that preserves {percent_variance} variance')
        fit_pca = pca.transform(self.new_features)[:, :nth_feature]

        kmeans_pca = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=40)
        kmeans_pca.fit(fit_pca)

        pca_df = pd.concat([self.new_df.reset_index(drop=True), pd.DataFrame(fit_pca[:, :3])], axis=1)
        pca_df.columns.values[-3: ] = ['pca_comp1', 'pca_comp2', 'pca_comp3']
        pca_df['k_means_cluster_id'] = kmeans_pca.labels_

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(kmeans_pca.labels_))]
        # with kmeans labelling
        x_axis = pca_df['pca_comp1']
        y_axis = pca_df['pca_comp2']
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x_axis, y_axis, hue=pca_df['k_means_cluster_id'], palette=sns.color_palette("hls", 8))
        plt.title('Clusters by PCA components and Kmeans with Kmeans labels')

        plt.autoscale(True)
        plt.savefig("mb_kmeans_kmeans_labels.png")
        if self.visualize:
            plt.show()

        # with gold labels of question types
        x_axis = pca_df['pca_comp1']
        y_axis = pca_df['pca_comp2']
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x_axis, y_axis, hue=pca_df['type'], palette=sns.color_palette("hls", 8))
        plt.title('Clusters by PCA components and Kmeans with question type labels')

        plt.autoscale(True)
        plt.savefig("mb_kmeans_qtype_labels.png")
        if self.visualize:
            plt.show()

    def tsne(self):
        qs = {}
        for q, ty in zip(self.new_df["question"].values, self.new_df["type"].values):
            if ty not in qs:
                qs[ty] = []
            qs[ty].append(q)

        ty_id_map = {ty: id for id, ty in enumerate(qs)}
        inv_ty_id_map = {ty_id_map[x]: x for x in ty_id_map}

        print(f'features shape: {self.new_features.shape}, subset_df size: {len(self.new_df)}')
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(self.new_features)

        self.new_df['tsne_x0'] = tsne_results[:, 0]
        self.new_df['tsne_x1'] = tsne_results[:, 1]

        plt.figure(figsize=(16, 10))
        
        sns.scatterplot(
            x='tsne_x0', y='tsne_x1', hue="type", palette=sns.color_palette("hls", 8),
            data=self.new_df, legend="full",)

        plt.savefig("video_tsne.png", dpi=120)
        if self.visualize:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--features_path", type=str, default='app_mot_train.h5')
    parser.add_argument("--csv_path", type=str, default="train.csv")
    parser.add_argument("--csv_answers_key", type=str, default="answer")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--ques_type_key", type=str, default="type")
    parser.add_argument("--ques_type", type=str, default=None)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--transform_type", type=str, choices=["all", "one", "avg"], default="avg")
    parser.add_argument("--feature_type", type=str, choices=["original", "c3d", "vit"], default="vit")
    parser.add_argument("--frame_number", type=int, default=0)
    # Clustering args
    # KMeans
    parser.add_argument("--test_clusters", type=int, default=11)
    parser.add_argument("--elbow_plot", type=str, default="num_cluster_kmeans_100.png")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--kmeans_savefile", type=str, default="kmeans.png")

    args = parser.parse_args()

    _data = Data(features_path=args.features_path, csv_path=args.csv_path, slice=args.slice, \
                feature_type=args.feature_type, transform_type=args.transform_type, frame_number=args.frame_number)

    new_features, new_df = _data.get_features_and_subset_df()

    _clustering = Clustering(new_features, new_df, args.visualize)

    # KMEANS
    _clustering.mb_kmeans(n_clusters=args.n_clusters)

    # tSNE
    _clustering.tsne()
