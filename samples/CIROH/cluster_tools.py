import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import umap
from minisom import MiniSom
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


class ClusterCollection:
    def __init__(self, subs=None):
        print('Initializing Cluster Collection')
        self.feature_data = None
        self.X = None
        if subs is not None:
            self.subs = subs
        else:
            self.subs = list()

    def compile(self):
        self.feature_data = pd.concat([s.feature_data for s in self.subs])
        self.X = np.concatenate([s.X for s in self.subs])
        self.medoid_dict = dict()
        for s in self.subs:
            self.medoid_dict.update(s.medoid_dict)

class Clusterer:
    colors = ["#FF5733", "#FFBD33", "#FF3381", "#33FFC8", "#3364FF", "#FF3364", "#33FF57", "#33C8FF", "#33FFBD", "#64FF33", "#BD33FF", "#FFC833", "#FF33BD", "#C8FF33", "#57FF33", "#FF33C8"]

    def __init__(self, feature_data):
        print('Initializing Clusterer')
        self.feature_data = feature_data
        self.feature_data['cluster'] = np.nan
        self.feature_data['ignore'] = False
        self.feature_cols = None
        self.clusterer = None
        self.embedding = None
        self.X = None
        self.sub_clusters = dict()
        self.cpal = {i: self.colors[i] for i in range(len(self.colors))}
        
    def preprocess_features(self, feature_cols, norm_type='min-max'):
        self.feature_cols = feature_cols
        self.x_inds = self.feature_data.index[~self.feature_data['ignore']]
        X = self.feature_data.loc[self.x_inds, feature_cols].values

        # Remove skew
        init_floor = X.min(axis=0)
        X = X - init_floor
        skew = self.feature_data[feature_cols].skew()
        skewed_features = abs(skew) > 1.5
        for i, s in enumerate(skewed_features):
            if s:
                X[:, i][X[:, i] <= 0] = X[:, i][X[:, i] > 0].min()
        X[:, skewed_features] = np.log(X[:, skewed_features])

        # normalize
        x_means = X.mean(axis=0)
        x_stdevs = X.std(axis=0)
        x_min = X.min(axis=0)
        x_range = X.max(axis=0) - x_min
        if norm_type == 'standard':
            self.X = (X - x_means) / x_stdevs
        elif norm_type == 'min-max':
            self.X = (X - x_min) / x_range

        trans_dict = {
            'features': feature_cols,
            'init_floor': init_floor,
            'skew_cols': skewed_features,
            'norm_type': norm_type,
            'x_range': x_range,
            'x_min': x_min,
            'x_means': x_means,
            'x_stdevs': x_stdevs
        }
        self.trans_dict = trans_dict
    
    def preprocess_features_fromdict(self, trans_dict):
        self.feature_cols = trans_dict['features']
        X = self.feature_data[self.feature_cols].values

        # Remove skew
        X = X - trans_dict['init_floor']
        skewed_features = trans_dict['skew_cols']
        for i, s in enumerate(skewed_features):
            if s:
                X[:, i][X[:, i] <= 0] = X[:, i][X[:, i] > 0].min()
        X[:, skewed_features] = np.log(X[:, skewed_features])

        # normalize
        if trans_dict['norm_type'] == 'standard':
            self.X = (X - trans_dict['x_means']) / trans_dict['x_stdevs']
        elif trans_dict['norm_type'] == 'min-max':
            self.X = (X - trans_dict['x_min']) / trans_dict['x_range']

    def cluster(self):
        labels = self.clusterer.fit_predict(self.X)
        self.feature_data['cluster'] = labels
        self.find_medoids()

    def find_medoids(self):
        medoid_dict = dict()
        for c in self.feature_data['cluster'].unique():
            mask = self.feature_data['cluster'] == c
            subset = self.X[mask]
            subset = np.nan_to_num(subset)
            dists = pairwise_distances(subset, subset)
            medoid = np.argmin(dists.sum(axis=0))
            subset = self.feature_data.index[mask]
            medoid = subset[medoid]
            medoid_dict[c] = medoid
        self.medoid_dict = medoid_dict

    def calc_embedding(self, method='umap', sigma=2, lr=6, epochs=5000):
        print('Calculating Embeddings')
        if method == 'umap':
            reducer = umap.UMAP()
            self.embedding = reducer.fit_transform(self.X)
        elif method == 'som':
            pca = PCA(n_components=2)
            pca.fit(self.X)
            rat = (pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]) + 1
            y_dim = (5 * np.sqrt(len(self.X))) / rat
            x_dim = y_dim * rat
            y_dim = int(y_dim)
            x_dim = int(x_dim)
            som = MiniSom(x_dim, y_dim, self.X.shape[1], sigma=sigma, learning_rate=lr, random_seed=0)
            som.pca_weights_init(self.X)
            som.train(self.X, epochs)
            self.embedding = np.array([som.winner(x) for x in self.X])
            # add jitter
            self.embedding = self.embedding + (np.random.random(self.embedding.shape) - 0.5)
        elif method == 'pca':
            pca = PCA(n_components=2)
            pca.fit(self.X)
            self.embedding = pca.transform(self.X)
        elif method == 'tsne':
            tsne = TSNE(n_components=2, perplexity=30, n_iter=5000)
            self.embedding = tsne.fit_transform(self.X)

    def elbow_plot(self, max_bins=20):
        og_bins = self.clusterer.n_clusters
        bin_range = np.arange(2, max_bins)
        inertias = np.zeros(len(bin_range))
        ch_scores = np.zeros(len(bin_range))
        sil_scores = np.zeros(len(bin_range))
        for i, b in enumerate(bin_range):
            self.clusterer.n_clusters = b
            self.cluster()
            inertias[i] = self.clusterer.inertia_
            ch_scores[i] = calinski_harabasz_score(self.X, self.feature_data['cluster'])
            sil_scores[i] = silhouette_score(self.X, self.feature_data['cluster'])
        
        fig, axs = plt.subplots(nrows=3, figsize=(6, 9), sharex=True)
        axs[0].plot(bin_range, inertias)
        axs[0].set_ylabel('Inertia')
        axs[1].plot(bin_range, sil_scores)
        axs[1].set_ylabel('Silhouette Score')
        axs[2].plot(bin_range, ch_scores)
        axs[2].set_ylabel('Calinski-Harabasz Score')
        axs[2].set_xticks(bin_range)
        axs[2].set_xlabel('Number of Clusters')
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'elbow_plot.png'), dpi=300)
        self.clusterer.n_clusters = og_bins
    
    def vis_clusters(self):
        if self.embedding is None:
            self.calc_embedding()
        fig, ax = plt.subplots()
        cmap = ListedColormap(self.colors[:int(self.feature_data['cluster'].max() + 1)])
        cbar = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.feature_data['cluster'].to_numpy(), alpha=0.7, s=3, cmap=cmap)
        cbar = fig.colorbar(cbar)
        cbar.set_ticks(range(len(self.feature_data['cluster'].unique())))
        cbar.set_label('Cluster')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(self.out_dir, 'cluster_map.png'), dpi=300)

    def plot_features(self):
        transformed_df = pd.DataFrame(self.X, columns=self.feature_cols)
        out_dir = os.path.join(self.out_dir, 'feature_plots')
        os.makedirs(out_dir, exist_ok=True)
        for c in self.feature_cols:
            fig, axs = plt.subplots(nrows=2, figsize=(6.5, 6), sharey=True)
            sns.histplot(x=c, data=self.feature_data, ax=axs[0])
            sns.histplot(x=c, data=transformed_df, ax=axs[1])
            fig.savefig(os.path.join(out_dir, f'{c}_hist.png'), dpi=300)

    def plot_feature_boxplots(self):
        transformed_df = pd.DataFrame(self.X, columns=self.feature_cols)
        transformed_df['cluster'] = self.feature_data['cluster'].to_numpy()
        ord = sorted(transformed_df['cluster'].unique())

        cols = int(np.ceil(np.sqrt(len(self.feature_cols))))
        rows = int(len(self.feature_cols) / cols) + 1
        cols = 3
        rows = 3

        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(13, 9), sharey=True, sharex=True)
        axs[2, 0].remove()
        axs[2, 2].remove()
        ax_list = [ax for ax in axs.flat if ax.axes is not None]
        for i, ax in enumerate(ax_list):
            c = self.feature_cols[i]
            sns.boxplot(x='cluster', y=c, data=transformed_df, ax=ax, palette=self.cpal, order=ord)
            ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'feature_boxplots.png'), dpi=300)

    def plot_boxplots_general(self, col_list):
        cols = len(col_list)
        rows = 1
        ord = sorted(self.feature_data['cluster'].unique())

        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(4.33 * cols, 2.66 * rows), sharex=True)
        i = 0
        for c in col_list:
            sns.boxplot(x='cluster', y=c, data=self.feature_data, ax=axs[i], palette=self.cpal, order=ord)
            if self.feature_data[c].skew() > 1:
                axs[i].set_yscale('log')
            axs[i].set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5')
            i += 1

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'misc_boxplots.png'), dpi=300)


class SubClusterer(Clusterer):

    def __init__(self, parent):
        print('Making Sub-clusterer')
        self.parent = parent
        self.update_clusters = self.update_parent_clusters

    def update_parent_clusters(self):
        self.parent.update_clusters(self.feature_data.index, self.feature_data['cluster'])

