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

class Data():
    def __init__(self, features_path='app_mot_train.h5', transform_data=True, csv_path="train.csv", frame_number=0, id_key="ids", features_key="features", csv_id_key="video", all_frames=False):
        self.features_path = features_path
        self.id_key=id_key
        self.features_key=features_key
        self.csv_id_key=csv_id_key
        self.df = pd.read_csv(csv_path)
        self.ids, self.features = self.load_data(self.features_path, self.id_key, self.features_key, self.csv_id_key)
        self.frame_number=frame_number
        if transform_data:
            self.features = self.transform(self.features, self.frame_number, all_frames)
        
    def get_features(self):
        return self.features

    def load_data(self, features_path='app_mot_train.h5', id_key="ids", features_key="features", csv_id_key="video"):
        with h5py.File(features_path, 'r') as f:
            # (3870,) # (3870, 16, 4096)
            ids, features = f[id_key][()], f[features_key][()]
            
        vid_ids = self.df[csv_id_key].to_numpy()
        match = np.sum(np.isin(ids, vid_ids))
        assert match==len(ids), "All Ids not matching"
        return ids, features

    def transform(self, features, frame_number=0, all_frames=False):
        if all_frames:
            new_features = features.reshape(-1, features.shape[-1])
        else:
            new_features = features[:, frame_number, :]
        return new_features


# class Data(Dataset):
#     def __init__(self, features_path='app_mot_train.h5', transform_data=False, csv_path="train.csv", frame_number=0, id_key="ids", features_key="features", csv_id_key="video"):
#         self.features_path = features_path
#         self.id_key=id_key
#         self.features_key=features_key
#         self.csv_id_key=csv_id_key
#         self.df = pd.read_csv(csv_path)
#         self.features, self.ids = self.load_data(self.features_path, self.id_key, self.features_key, self.csv_id_key)
#         self.frame_number=frame_number
#         if transform_data:
#             self.features = self.transform(features=self.features)
        

#     def load_data(self, features_path='app_mot_train.h5', id_key="ids", features_key="features", csv_id_key="video"):
#         with h5py.File(features_path, 'r') as f:
#             # (3870,) # (3870, 16, 4096)
#             ids, features = f[id_key][()], f[features_key][()]
            
#         vid_ids = self.df[csv_id_key].to_numpy()
#         match = np.sum(np.isin(ids, vid_ids))
#         assert match==len(ids), "All Ids not matching"
#         return ids, features

#     def transform(self, features):
#         new_features = features.reshape(-1, features.shape[-1])
#         return new_features
    
#     def __getitem__(self, index):
#         return self.features[index][self.frame_number], index


class Clustering:
    def __init__(self, new_features, visualize):
        self.new_features = new_features
        self.colors_ = cycle(colors.cnames.keys())
        self.n_clusters = 5
        self.visualize = visualize
    
    # SOURCE: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    def dbscan(self, eps=1, min_samples=16, filename="dbscan.png"):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(self.new_features)
        print("DBSCAN Fit Done")
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.new_features[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = self.new_features[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.savefig(filename)
        if self.visualize:
            plt.show()

    # SOURCE: https://scikit-learn.org/stable/auto_examples/cluster/plot_birch_vs_minibatchkmeans.html
    def birch(self, threshold=1.7, n_clusters=None, filename="birch.png"):
        # Use all colors that matplotlib provides by default.
        fig = plt.figure(figsize=(12, 4))
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

        # Compute clustering with BIRCH with and without the final clustering step and plot.
        birch_models = [
            Birch(threshold=threshold, n_clusters=n_clusters),
            Birch(threshold=1.7, n_clusters=5),
        ]

        final_step = ["without global clustering", "with global clustering"]

        for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
            birch_model.fit(self.new_features)

            # Plot result
            labels = birch_model.labels_
            centroids = birch_model.subcluster_centers_
            n_clusters = np.unique(labels).size
            print("n_clusters : %d" % n_clusters)

            ax = fig.add_subplot(1, 2, ind + 1)
            for this_centroid, k, col in zip(centroids, range(n_clusters), self.colors_):
                mask = labels == k
                ax.scatter(self.new_features[mask, 0], self.new_features[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.5)
                if birch_model.n_clusters is None:
                    ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)
            ax.set_autoscaley_on(True)
            ax.set_title("BIRCH %s" % info)

        plt.savefig(filename)
        if self.visualize:
            plt.show()


    def kmeans_num_clusters(self, num_clusters=101, filename="num_cluster_kmeans_100.png"):
        WCSS = []
        for i in range(1, num_clusters):
            print(i)
            model = MiniBatchKMeans(n_clusters = i, init = 'k-means++',batch_size=256, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
            model.fit(self.new_features)
            WCSS.append(model.inertia_)
        fig = plt.figure(figsize = (7,7))
        plt.plot(range(1, num_clusters), WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
        plt.xticks(np.arange(0, num_clusters, step=5))
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.savefig(filename)

    # SOURCE: https://scikit-learn.org/stable/auto_examples/cluster/plot_birch_vs_minibatchkmeans.html
    def mb_kmeans(self, n_clusters):
        mbk = MiniBatchKMeans(
            init="k-means++",
            n_clusters=n_clusters,
            batch_size=256,
            n_init=10,
            max_no_improvement=10,
            verbose=0,
            random_state=0,
        )

        mbk.fit(self.new_features)
        print("KMEANS Fit Done")
        mbk_means_labels_unique = np.unique(mbk.labels_)

        for this_centroid, k, col in zip(mbk.cluster_centers_, range(n_clusters), ["r", "b"]):
            mask = mbk.labels_ == k
            plt.scatter(self.new_features[mask, 0], self.new_features[mask, 1], marker=".", c="white", edgecolor=col, alpha=0.5)
            plt.scatter(this_centroid[0], this_centroid[1], marker="+", c="black", s=25)
        plt.title("MiniBatchKMeans")
        plt.autoscale(True)
        plt.savefig("mb_kmeans.png")
        if self.visualize:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--frame_number", type=int, default=0)
    parser.add_argument("--features_path", type=str, default='app_mot_train.h5')
    parser.add_argument("--transform_data", type=bool, default=True)
    parser.add_argument("--csv_path", type=str, default="train.csv")
    parser.add_argument("--id_key", type=str, default="ids")
    parser.add_argument("--features_key", type=str, default="feat")
    parser.add_argument("--csv_id_key", type=str, default="video")
    parser.add_argument("--all_frames", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=True)

    # Clustering args
    # DBSCAN
    parser.add_argument("--eps", type=float, default=100)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--dbscan_savefile", type=str, default="dbscan.png")
    # KMeans
    parser.add_argument("--test_clusters", type=int, default=11)
    parser.add_argument("--elbow_plot", type=str, default="num_cluster_kmeans_100.png")
    parser.add_argument("--n_clusters", type=int, default=1)
    parser.add_argument("--kmeans_savefile", type=str, default="kmeans.png")
    # BIRCH
    parser.add_argument("--threshold", type=float, default=1.7)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--birch_savefile", type=str, default="birch.png")

    args = parser.parse_args()
    
    _data = Data(features_path=args.features_path, transform_data=args.transform_data, csv_path=args.csv_path, frame_number=args.frame_number, \
                id_key=args.id_key, features_key=args.features_key, csv_id_key=args.csv_id_key, all_frames=args.all_frames)
    new_features = _data.get_features()

    _clustering = Clustering(new_features, args.visualize)
    # DBSCAN
    # _clustering.dbscan(eps=args.eps, min_samples=args.min_samples, filename=args.dbscan_savefile)
    # KMEANS
    # _clustering.kmeans_num_clusters(num_clusters=args.test_clusters, filename=args.elbow_plot)
    _clustering.mb_kmeans(n_clusters=args.n_clusters)
    # BIRCH
    # _clustering.birch(threshold=args.threshold, n_clusters=args.num_clusters)
    
