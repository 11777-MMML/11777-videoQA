from pyexpat import features
import h5py
import pandas as pd
import numpy as np
from itertools import cycle
from sklearn.cluster import Birch, MiniBatchKMeans, DBSCAN, KMeans
from sklearn import metrics
import matplotlib.colors as colors
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class Data:
    def __init__(self, features_path='bert_ft_train.h5', csv_path='train.csv', csv_answers_key='answer', slice=None, ques_type_key='type', ques_type=None):
        self.features_path = features_path
        self.df = pd.read_csv(csv_path)
        self.csv_answers_key = csv_answers_key
        self.slice = slice
        self.ques_type = ques_type
        self.subset_index = []
        self.ques_type_key = ques_type_key
        self.features, self.subset_df = self.load_data(self.features_path, self.csv_answers_key, self.slice, self.ques_type_key, self.ques_type)

    def get_features_and_subset_df(self):
        return self.features, self.subset_df

    def load_data(self, features_path='bert_ft_train.h5', csv_answers_key='answer', slice=None, ques_type_key='type', ques_type=None):
        with h5py.File(features_path, 'r') as f:
            feats = f['feat']
            if slice is not None:
                feats = feats[:slice]
                self.df = self.df[:slice]

            if ques_type is not None:
                self.subset_index = self.df[self.df[ques_type_key] == ques_type].index
                feats = feats[self.subset_index]

        if len(self.subset_index) == 0:
            feats = feats[np.arange(len(feats)), self.df[csv_answers_key].tolist(), :, :]
        else:
            feats = feats[np.arange(len(feats)), (self.df[self.subset_index][csv_answers_key]).tolist(), :, :]
        qa_lens = [self.nozero_row(feats[i]) for i in range(0, len(feats))]
        feats = [np.mean(feats[i, :qa_lens[i], :], axis=0) for i in range(0,len(feats))]
        return np.array(feats), self.df

    def nozero_row(self, A):
        i = 0
        for row in A:
            if row.sum() == 0:
                break
            i += 1

        return i


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

        kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=40)
        kmeans_pca.fit(fit_pca)

        pca_df = pd.concat([self.new_df.reset_index(drop=True), pd.DataFrame(fit_pca[:, :3])], axis=1)
        pca_df.columns.values[-3: ] = ['pca_comp1', 'pca_comp2', 'pca_comp3']
        pca_df['k_means_cluster_id'] = kmeans_pca.labels_

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

        print(f'features shape: {new_features.shape}, subset_df size: {len(new_df)}')
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(new_features)

        new_df['tsne_x0'] = tsne_results[:, 0]
        new_df['tsne_x1'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='tsne_x0', y='tsne_x1', hue='type', palette=sns.color_palette("hls", 8),
            data=new_df, legend="full",
            alpha=0.3
        )
        plt.legend(labels=[inv_ty_id_map[x] for x in range(8)])
        plt.savefig("qa_pair_tsne.png", dpi=120)
        if self.visualize:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--features_path", type=str, default='bert_ft_train.h5')
    parser.add_argument("--csv_path", type=str, default="train.csv")
    parser.add_argument("--csv_answers_key", type=str, default="answer")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--ques_type_key", type=str, default="type")
    parser.add_argument("--ques_type", type=str, default=None)
    parser.add_argument("--visualize", type=bool, default=True)

    # Clustering args
    # KMeans
    parser.add_argument("--test_clusters", type=int, default=11)
    parser.add_argument("--elbow_plot", type=str, default="num_cluster_kmeans_100.png")
    parser.add_argument("--n_clusters", type=int, default=1)
    parser.add_argument("--kmeans_savefile", type=str, default="kmeans.png")

    args = parser.parse_args()

    _data = Data(features_path=args.features_path, csv_path=args.csv_path, csv_answers_key=args.csv_answers_key, slice=args.slice, ques_type_key=args.ques_type_key, ques_type=args.ques_type)
    new_features, new_df = _data.get_features_and_subset_df()

    _clustering = Clustering(new_features, new_df, args.visualize)
    # KMEANS
    _clustering.mb_kmeans(n_clusters=args.n_clusters)

    # tSNE
    # _clustering.tsne()


