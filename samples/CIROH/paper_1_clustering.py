import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MeanShift, SpectralClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score, pairwise_distances
from sklearn.manifold import TSNE
import umap
from minisom import MiniSom
import statsmodels.api as sm
from scipy.optimize import minimize
import json
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import itertools as it
import copy


class Clusterer:

    def __init__(self, run_path):
        print('Loading data')
        with open(run_path, 'r') as f:
            run_dict = json.loads(f.read())
        self.run_dict = run_dict
        self.out_dir = os.path.join(run_dict['analysis_directory'], 'clustering')
        os.makedirs(self.out_dir, exist_ok=True)
        working_dir = run_dict['geometry_directory']
        features_path = run_dict['analysis_path']
        el_path = os.path.join(working_dir, 'el.csv')
        el_scaled_path = os.path.join(working_dir, 'el_scaled.csv')
        rh_prime_path = os.path.join(working_dir, 'rh_prime.csv')
        width_path = os.path.join(working_dir, 'area.csv')
        celerity_path = os.path.join(working_dir, 'celerity.csv')

        feature_data = pd.read_csv(features_path)
        # feature_data = feature_data.dropna(axis=0)
        feature_data = feature_data[feature_data['invalid_geometry'] == 0].copy()
        feature_data[run_dict['id_field']] = feature_data[run_dict['id_field']].astype(np.int64).astype(str)
        self.feature_data = feature_data.set_index(run_dict['id_field'])

        el_data = pd.read_csv(el_path)
        self.el_data = el_data.dropna(axis=1)

        el_scaled_data = pd.read_csv(el_scaled_path)
        self.el_scaled_data = el_scaled_data.dropna(axis=1)

        width_data = pd.read_csv(width_path)
        self.width_data = width_data.dropna(axis=1)

        cel_data = pd.read_csv(celerity_path)
        self.cel_data = cel_data.dropna(axis=1)
        self.feature_data['celerity'] = 0.0
        for c in cel_data.columns:
            if c in self.feature_data.index:
                tmp_el = el_scaled_data[c].values
                edap = self.feature_data.loc[c, 'el_edap_scaled']
                edap = np.argmax(tmp_el > edap)
                edep = self.feature_data.loc[c, 'el_edep_scaled']
                edep = np.argmax(tmp_el > edep)
                if edap >= edep:
                    edep = edap + 1
                self.feature_data.loc[c, 'celerity'] = cel_data[c].values[edap:edep].mean()  # average celerity for EDZ
            else:
                self.feature_data.loc[c, 'celerity'] = cel_data[c].values[0:500].mean()
        self.feature_data['celerity_detrended'] = np.log(self.feature_data['celerity']) - np.log((self.feature_data['slope'] ** 0.5) * (1 / 0.07))
        self.feature_data['celerity_detrended'] = np.exp(self.feature_data['celerity_detrended'])

        rh_prime_data = pd.read_csv(rh_prime_path)
        # Clean Rh prime
        rh_prime_data.iloc[-1] = rh_prime_data.iloc[-2]
        rh_prime_data[:] = gaussian_filter1d(rh_prime_data.T, 15).T
        rh_prime_data[rh_prime_data < -3] = -3
        self.rh_prime_data = rh_prime_data.dropna(axis=1)

        # Add attenuation
        if os.path.exists(run_dict['muskingum_path']):
            magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
            durations = ['Short', 'Medium', 'Long']

            with open(r'source/regressions.json', 'r') as f:
                regressions = json.loads(f.read())
            regressions = regressions['peak_flowrate']
            for m in magnitudes:
                peak_estimate = ((self.feature_data['DASqKm'].to_numpy() / 2.59) ** regressions[m][1]) * regressions[m][0] * (1 / 35.3147)
                for d in durations:
                    self.feature_data[f'{m}_{d}_cms_attenuation'] = self.feature_data[f'{m}_{d}_pct_attenuation'] * peak_estimate
                    total_lengths = (self.feature_data[f'{m}_{d}_dx'] * self.feature_data[f'{m}_{d}_subreaches']) / 1000
                    self.feature_data[f'{m}_{d}_cms_attenuation_per_km'] = self.feature_data[f'{m}_{d}_cms_attenuation'] / total_lengths

        self.clusterer = None
        self.embedding = None
        self.X = None

        self.cpal = {'A': '#fdc161',
                     'B': '#c1bae0',
                     'C': '#badeab',
                     'D': '#fbabb7', 
                     'E': '#03bde3', 
                     'F': '#fbf493',
                     'G': '#a3d4e4',
                     'H': '#f9d5e5',
                     'I': '#b6e3a8'}
        self.cpal = {'A': '#FF5733',
                     'B': '#FFC300',
                     'C': '#219c21',
                     'D': '#3366FF', 
                     'E1': '#FF33EA', 
                     'E2': '#15e8d2',
                     'G': '#FF3366',
                     'H': '#CC33FF',
                     'I': '#33CCFF',
                     'X': '#807e7e'}

    def vc_removal(self, vc_cutoff):
        l1 = len(self.feature_data)
        self.feature_data = self.feature_data.query(f'valley_confinement > {vc_cutoff}').copy()
        print(f'Removed {l1 - len(self.feature_data)} reaches for valley confinement')
    
    def slope_removal(self, slope_cutoff):
        l1 = len(self.feature_data)
        self.feature_data = self.feature_data.query(f'slope < {slope_cutoff}').copy()
        print(f'Removed {l1 - len(self.feature_data)} reaches for slope')
    
    def wbody_removal(self):
        l1 = len(self.feature_data)
        self.feature_data = self.feature_data.query('wbody != 1').copy()
        print(f'Removed {l1 - len(self.feature_data)} reaches for wbody')

    def preprocess_features(self, feature_cols, norm_type='min-max'):
        self.feature_cols = feature_cols
        X = self.feature_data[feature_cols].values

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
        return trans_dict
    
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
        self.feature_data.loc[:, 'cluster'] = labels

        # find median item of each cluster
        medoid_dict = dict()
        for c in np.unique(labels):
            subset = self.X[labels == c]
            dists = pairwise_distances(subset, subset)
            medoid = np.argmin(dists.sum(axis=0))
            subset = self.feature_data.index[labels == c]
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

    def optimize_som(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        rat = (pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]) + 1
        y_dim = (5 * np.sqrt(len(self.X))) / rat
        x_dim = y_dim * rat
        y_dim = int(y_dim)
        x_dim = int(x_dim)

        som = MiniSom(x_dim, y_dim, self.X.shape[1], sigma=1, learning_rate=1, random_seed=0)
        som.pca_weights_init(self.X)
        init_weights = som.get_weights().copy()

        scalars = [5, 4, 500]

        def som_loss(params, weights=init_weights):
            sigma = params[0] * scalars[0]
            lr = params[1] * scalars[1]
            epochs = int(params[2] * scalars[2])
            som = MiniSom(x_dim, y_dim, self.X.shape[1], sigma=sigma, learning_rate=lr)
            som._weights = weights
            som.train(self.X, epochs)
            return som.quantization_error(self.X)
        
        res = minimize(som_loss, [0.5, 0.5, 0.5], bounds=[(0.1, 1), (0.1, 1), (0.1, 1)])
        print(f'Found Optimal SOM params: {res.x[0] * scalars[0]}, {res.x[1] * scalars[1]}, {int(res.x[2] * scalars[2])}')

    def som_param_sweep(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        rat = (pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]) + 1
        y_dim = (5 * np.sqrt(len(self.X))) / rat
        x_dim = y_dim * rat
        y_dim = int(y_dim)
        x_dim = int(x_dim)

        som = MiniSom(x_dim, y_dim, self.X.shape[1], sigma=1, learning_rate=1, random_seed=0)
        som.pca_weights_init(self.X)
        init_weights = som.get_weights().copy()

        mins = [1.9, 5, 1500]
        ranges = [0.5, 2.5, 3000]

        def som_loss(params, weights=init_weights):
            sigma = (params[0] * ranges[0]) + mins[0]
            lr = (params[1] * ranges[1]) + mins[1]
            epochs = int((params[2] * ranges[2])) + mins[2]
            som = MiniSom(x_dim, y_dim, self.X.shape[1], sigma=sigma, learning_rate=lr)
            som._weights = weights
            som.train(self.X, epochs)
            return som.quantization_error(self.X)
        
        grid_res = 4
        param_grid = np.meshgrid(np.linspace(0.1, 1, grid_res), np.linspace(0.1, 1, grid_res), np.linspace(0.1, 1, grid_res))
        param_grid = np.array(param_grid).reshape(3, -1).T
        losses = np.zeros(param_grid.shape[0])
        counter = 1
        for i, p in enumerate(param_grid):
            print(counter)
            counter += 1
            losses[i] = som_loss(p)
        
        # print argmin values
        min_idx = np.argmin(losses)
        opt_params = [param_grid[min_idx][0] * ranges[0], param_grid[min_idx][1] * ranges[1], int(param_grid[min_idx][2] * ranges[2])]
        opt_params = [opt_params[0] + mins[0], opt_params[1] + mins[1], opt_params[2] + mins[2]]
        print(f'Found Optimal SOM params: {opt_params}')

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

    def vis_celerity(self, detrended=False, plot_name='celerity_map.png'):
        if self.embedding is None:
            self.calc_embedding()
        if detrended:
            c = self.feature_data['celerity_detrended'].to_numpy()
        else:
            c = self.feature_data['celerity'].to_numpy()
        fig, ax = plt.subplots()
        cbar = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c=c, alpha=0.7, s=3, cmap='plasma')
        cbar = fig.colorbar(cbar)
        if detrended:
            cbar.set_label('Detrended Celerity')
        else:
            cbar.set_label('Celerity')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(self.out_dir, plot_name), dpi=300)
    
    def vis_clusters(self):
        if self.embedding is None:
            self.calc_embedding()
        fig, ax = plt.subplots()
        color_list = ['#fdc161','#c1bae0','#badeab', '#03bde3', '#fbabb7', '#fbf493']
        color_list = ['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF']
        cmap = ListedColormap(color_list[:int(self.feature_data['cluster'].max() + 1)])
        cbar = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.feature_data['cluster'].to_numpy(), alpha=0.7, s=3, cmap=cmap)
        cbar = fig.colorbar(cbar)
        cbar.set_ticks(range(len(self.feature_data['cluster'].unique())))
        cbar.set_label('Cluster')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(self.out_dir, 'cluster_map.png'), dpi=300)

    def _plot_rhp(self, ax):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        y_space = np.linspace(0, 6, self.rh_prime_data.shape[0])

        for i in range(len(n_clusters)):
            mask = self.feature_data['cluster'] == ord[i]

            # trace each rh' curve within the label
            for r in self.feature_data[mask].iterrows():
                reach = r[0]
                rh_prime = self.rh_prime_data[reach].values
                if ord[i] not in self.cpal:
                    c = 'k'
                else:
                    c = self.cpal[ord[i]]
                ax[i].plot(rh_prime, y_space, c=c, alpha=0.1)
            
            # Formatting and labelling
            ax[i].set_title(f'n={len(self.feature_data[mask])}')
            ax[i].set_xlabel(r"${R}_{h}$'")
            ax[i].set_xlim(-3, 1)
            ax[i].set_yticks([])
            ax[i].set_facecolor('#f5f5f5')
        
        ax[0].set_ylabel('Stage (x bkf)')
        ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6])
        return ax
    
    def _plot_sec(self, ax):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        stage_space = np.linspace(0, 6, self.width_data.shape[0])
        stage_space = np.append(stage_space[::-1], stage_space)

        max_w = 400
        new_ws = dict()
        for r in self.feature_data.iterrows():
            length = r[1]['length']
            r = r[0]
            widths = self.width_data[r].values / length
            widths = np.append(-widths[::-1], widths)
            widths = widths / 2
            new_ws[r] = widths

       # plot sections
        for i in range(len(n_clusters)):
            mask = self.feature_data['cluster'] == ord[i]
            for r in self.feature_data[mask].iterrows():
                r = r[0]
                widths = new_ws[r]
                widths = widths + max_w
                if ord[i] not in self.cpal:
                    c = 'k'
                else:
                    c = self.cpal[ord[i]]
                ax[i].plot(widths, stage_space, c=c, alpha=0.1)

            ax[i].set_xlim(0, max_w * 2)
            ax[i].set_xlabel('Top-width (m)')
            ax[i].set_xticks([0, max_w * 2])
            ax[i].set_facecolor('#f5f5f5')

        return ax

    def _plot_sec_medoid(self, ax, med_reach_dict):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        stage_space = np.linspace(0, 6, self.width_data.shape[0])
        stage_space = np.append(stage_space[::-1], stage_space)

        new_ws = dict()
        for r in self.feature_data.iterrows():
            length = r[1]['length']
            r = r[0]
            widths = self.width_data[r].values / length
            widths = np.append(-widths[::-1], widths)
            widths = widths / 2
            new_ws[r] = widths

        # plot sections
        max_w = 400
        max_s = 4
        for i in range(len(n_clusters)):
            r = med_reach_dict[ord[i]]
            ax[i].set_xlim(0, max_w * 2)
            ax[i].set_xlabel('Top-width (m)')
            ax[i].set_xticks([0, max_w * 2])
            ax[i].set_yticks([])
            ax[i].set_facecolor('#f5f5f5')
            if r is None:
                continue
            widths = new_ws[r]
            widths = widths + max_w
            ax[i].plot(widths, stage_space, c='k', alpha=1)
            ax[i].set_ylim(0, max_s)
        ax[0].set_ylabel('Stage (x bkf)')
        ax[0].set_ylim(0, max_s)
        ax[0].set_yticks(range(max_s + 1))
        return ax

    def plot_cluster_hydraulics(self):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        fig, axs = plt.subplots(ncols=len(n_clusters), nrows=4, figsize=(3 * len(n_clusters), 16), gridspec_kw={'height_ratios': [4, 1, 1, 1]})

        rhp_axs = axs[0, :]
        rhp_axs = self._plot_rhp(rhp_axs)

        sec_axs = axs[1, :]
        sec_axs = self._plot_sec(sec_axs)

        # merge bottom row of subplots
        gs = axs[2, 0].get_gridspec()
        for ax in axs[2, :]:
            ax.remove()
        cel_ax = fig.add_subplot(gs[2, :])
        cel_ax.set_facecolor('#f5f5f5')
        sns.boxplot(x='cluster', y='celerity_detrended', data=self.feature_data, ax=cel_ax, showfliers=False, palette=self.cpal, order=ord)
        
        gs = axs[3, 0].get_gridspec()
        for ax in axs[3, :]:
            ax.remove()
        cel2_ax = fig.add_subplot(gs[3, :])
        cel2_ax.set_facecolor('#F5F5F5')
        sns.boxplot(x='cluster', y='celerity', data=self.feature_data, ax=cel2_ax, showfliers=False, palette=self.cpal, order=ord)
    
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'cluster_hydraulics.png'), dpi=300)

    def plot_cluster_hydraulics_2(self):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        fig, axs = plt.subplots(ncols=len(n_clusters), nrows=5, figsize=(19, 13), gridspec_kw={'height_ratios': [0.01, 3, 1, 1, 1]})

        for i in range(len(n_clusters)):
            axs[0, i].text(0.5, 0.5, f'{ord[i]}', fontsize=14, ha='center', va='center')
            axs[0, i].axis('off')
            

        rhp_axs = axs[1, :]
        rhp_axs = self._plot_rhp(rhp_axs)

        sec_axs = axs[2, :]
        sec_axs = self._plot_sec(sec_axs)

        # merge bottom row of subplots
        gs = axs[3, 0].get_gridspec()
        for ax in axs[3, :]:
            ax.remove()
        cel_ax = fig.add_subplot(gs[3, :])
        cel_ax.set_facecolor('#f5f5f5')
        sns.boxplot(x='cluster', y='celerity_detrended', data=self.feature_data, ax=cel_ax, showfliers=False, palette=self.cpal, order=ord)
        cel_ax.set(xlabel=None, ylabel='Shape Celerity', facecolor='#f5f5f5', xticks=[])
        
        gs = axs[4, 0].get_gridspec()
        for ax in axs[4, :]:
            ax.remove()
        cel2_ax = fig.add_subplot(gs[4, :])
        sns.boxplot(x='cluster', y='celerity', data=self.feature_data, ax=cel2_ax, showfliers=False, palette=self.cpal, order=ord)
        cel2_ax.set(xlabel=None, ylabel='Celerity (m/s)', facecolor='#f5f5f5', xticks=[])
    
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'cluster_hydraulics.png'), dpi=300)

    def plot_features(self):
        transformed_df = pd.DataFrame(self.X, columns=self.feature_cols)
        out_dir = os.path.join(self.out_dir, 'feature_plots')
        os.makedirs(out_dir, exist_ok=True)
        for c in self.feature_cols:
            fig, axs = plt.subplots(nrows=2, figsize=(6.5, 6), sharey=True)
            sns.histplot(x=c, data=self.feature_data, ax=axs[0])
            sns.histplot(x=c, data=transformed_df, ax=axs[1])
            fig.savefig(os.path.join(out_dir, f'{c}_hist.png'), dpi=300)

    def plot_edzs(self):
        n_clusters = self.feature_data['cluster'].astype(int).unique()
        fig, axs = plt.subplots(ncols=len(n_clusters), figsize=(3 * len(n_clusters), 3), sharey=True)
        
        for i in n_clusters:
            mask = self.feature_data['cluster'] == i

            # trace each rh' curve within the label
            for r in self.feature_data[mask].iterrows():
                reach = r[0]
                edap_scaled = self.feature_data.loc[reach, 'el_edap_scaled']
                el_scaled = self.el_scaled_data[reach].values
                rh_prime = self.rh_prime_data[reach].values
                start_ind = np.argmax(el_scaled > edap_scaled) - 1
                el_scaled = el_scaled - el_scaled[start_ind]

                axs[i].plot(rh_prime[start_ind:], el_scaled[start_ind:], c='b', alpha=0.1)
            
            # Formatting and labelling
            axs[i].set_title(f'n={len(self.feature_data[mask])}')
            axs[i].set_ylabel('Stage (m)')
            axs[i].set_xlabel(r"${R}_{h}$'")
            axs[i].set_xlim(-3, 1)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'edz_plot.png'), dpi=300)

    def plot_feature_boxplots(self):
        transformed_df = pd.DataFrame(self.X, columns=self.feature_cols)
        transformed_df['cluster'] = self.feature_data['cluster'].to_numpy()
        ord = sorted(transformed_df['cluster'].unique())

        cols = int(np.ceil(np.sqrt(len(self.feature_cols))))
        rows = int(len(self.feature_cols) / cols) + 1
        cols = 3
        rows = np.ceil(len(self.feature_cols) / cols).astype(int)
        
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(13, 9), sharey=True, sharex=True)
        if len(self.feature_cols) % cols == 1:
            axs[-1, 0].remove()
            axs[-1, 2].remove()
        elif len(self.feature_cols) % cols == 2:
            axs[-1, 2].remove()

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

    def plot_attenuation(self):
        magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
        durations = ['Short', 'Medium', 'Long']

        fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(13, 14.4), sharex=True, sharey='row')
        i = 0
        j = 0
        for t in ['pct_attenuation', 'pct_attenuation_per_km', 'cms_attenuation', 'cms_attenuation_per_km']:
            j = 0
            for d in durations:
                value_cols = [f'{m}_{d}_{t}' for m in magnitudes]
                melt = pd.melt(self.feature_data, id_vars='cluster', value_vars=value_cols, var_name='Event', value_name=t)
                melt[t] = melt[t].clip(lower=0)
                sns.boxplot(x='Event', y=t, hue='cluster', data=melt, ax=axs[i, j], palette=self.cpal, showfliers=False, hue_order=ord)
                b = axs[i, j].legend()
                b.remove()
                axs[i, j].set(xticks=range(len(magnitudes)), xticklabels=magnitudes)
                if i == 0:
                    axs[i, j].set(title=d)
                if i != 3:
                    axs[i, j].set(xlabel=None)
                if j != 0:
                    axs[i, j].set(ylabel=None)
                axs[i, j].set_facecolor('#f5f5f5')
                j += 1
            i += 1

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'attenuation_boxplots.png'), dpi=300)

    def plot_simple_hydraulics(self, med_reach_dict):
        n_clusters = self.feature_data['cluster'].unique()
        ord = sorted(n_clusters)
        fig, axs = plt.subplots(ncols=len(n_clusters), nrows=2, figsize=(13, 7), gridspec_kw={'height_ratios': [5, 1]})
            
        rhp_axs = axs[0, :]
        rhp_axs = self._plot_rhp(rhp_axs)

        w_axs = axs[1, :]
        w_axs = self._plot_sec(w_axs)
        w_axs = self._plot_sec_medoid(w_axs, med_reach_dict)
    
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'cluster_hydraulics_2.png'), dpi=300)

    def plot_celerity(self):
        ord = sorted(self.feature_data['cluster'].unique())
        # self.feature_data.loc[self.feature_data['cluster'] == 'A', 'celerity'] = np.nan
        # self.feature_data.loc[self.feature_data['cluster'] == 'A', 'celerity_detrended'] = np.nan

        fig, axs = plt.subplots(ncols=2, figsize=(13, 3.6))
        sns.boxplot(self.feature_data, x='cluster', y='celerity', ax=axs[0], palette=self.cpal, order=ord, showfliers=False)
        sns.boxplot(self.feature_data, x='cluster', y='celerity_detrended', ax=axs[1], palette=self.cpal, order=ord, showfliers=False)
        axs[0].set_facecolor('#f5f5f5')
        axs[1].set_facecolor('#f5f5f5')
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'celerity_plot.png'), dpi=300)

    def plot_routing(self):
        ord = sorted(self.feature_data['cluster'].unique())
        magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
        lables = ['50% AEP', '10% AEP', '2% AEP', '1% AEP']

        cms_label = 'Attenuation Per Km (cms)'
        value_cols = [f'{m}_Medium_cms_attenuation_per_km' for m in magnitudes]
        rename_dict = {i: j for i, j in zip(value_cols, lables)}
        cms_melt = pd.melt(self.feature_data, id_vars='cluster', value_vars=value_cols, var_name='Event', value_name=cms_label)
        cms_melt['Event'] = cms_melt['Event'].apply(lambda x: rename_dict[x])
        cms_melt[cms_label] = cms_melt[cms_label].clip(lower=0)

        pct_label = 'Attenuation Per Km (pct)'
        value_cols = [f'{m}_Medium_pct_attenuation_per_km' for m in magnitudes]
        rename_dict = {i: j for i, j in zip(value_cols, lables)}
        pct_melt = pd.melt(self.feature_data, id_vars='cluster', value_vars=value_cols, var_name='Event', value_name=pct_label)
        pct_melt['Event'] = pct_melt['Event'].apply(lambda x: rename_dict[x])
        pct_melt[pct_label] = pct_melt[pct_label].clip(lower=0)


        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 9))
        sns.boxplot(x='Event', y=cms_label, hue='cluster', data=cms_melt, ax=axs[0, 0], palette=self.cpal, showfliers=False, hue_order=ord)
        b = axs[0, 0].legend()
        b.remove()
        sns.boxplot(x='Event', y=pct_label, hue='cluster', data=pct_melt, ax=axs[0, 1], palette=self.cpal, showfliers=False, hue_order=ord)
        b = axs[0, 1].legend()
        b.remove()
        sns.boxplot(self.feature_data, x='cluster', y='Celerity (m/s)', ax=axs[1, 0], palette=self.cpal, order=ord, showfliers=False)
        sns.boxplot(self.feature_data, x='cluster', y='Shape Celerity (m^(2/3))', ax=axs[1, 1], palette=self.cpal, order=ord, showfliers=False)
        for i in range(2):
            for j in range(2):
                axs[i, j].set_facecolor('#f5f5f5')

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'routing_plot.png'), dpi=300)

def main2():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.feature_data['fp_slope'] = (clusterer.feature_data['w_edep'] - clusterer.feature_data['w_edap']) / clusterer.feature_data['height']
    clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] <= 0, 'fp_slope'] = clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] > 0, 'fp_slope'].min()
    clusterer.vc_removal(1.5)
    clusterer.slope_removal(10 ** -3)
    # features = ['w_edep', 'el_edep', 'w_edap', 'el_edap', 'fp_slope', 'valley_confinement', 'vol', 'min_rhp', 'cumulative_volume', 'cumulative_height', 'height', 'height_scaled', 'slope_min_stop']
    features = ['fp_slope', 'edz_count', 'w_edap', 'vol', 'height', 'el_edap', 'valley_confinement']
    clusterer.preprocess_features(features)
    clusterer.calc_embedding(method='umap')
    clusterer.feature_data['embed_x'] = clusterer.embedding[:, 0]
    clusterer.feature_data['embed_y'] = clusterer.embedding[:, 1]
    clusterer.preprocess_features(['embed_x', 'embed_y'])
    # clusterer.clusterer = KMeans(n_clusters=5, random_state=0)
    clusterer.clusterer = DBSCAN(eps=0.1, min_samples=5)
    clusterer.cluster()
    clusterer.preprocess_features(features)

    clusterer.vis_clusters()
    clusterer.plot_cluster_hydraulics()
    clusterer.vis_celerity()

def main3():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.feature_data['fp_slope'] = (clusterer.feature_data['w_edep'] - clusterer.feature_data['w_edap']) / clusterer.feature_data['height']
    clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] <= 0, 'fp_slope'] = clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] > 0, 'fp_slope'].min()
    clusterer.vc_removal(1.5)
    clusterer.slope_removal(10 ** -3)
    features = ['w_edep', 'el_edep', 'w_edap', 'el_edap', 'fp_slope', 'valley_confinement', 'vol', 'min_rhp', 'cumulative_volume', 'cumulative_height', 'height', 'height_scaled', 'slope_min_stop']
    clusterer.preprocess_features(features)
    # clusterer.som_param_sweep()
    # return
    clusterer.calc_embedding(method='som')
    clusterer.vis_celerity(detrended=True)

def main4():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.feature_data['fp_slope'] = (clusterer.feature_data['w_edep'] - clusterer.feature_data['w_edap']) / clusterer.feature_data['height']
    clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] <= 0, 'fp_slope'] = clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] > 0, 'fp_slope'].min()
    clusterer.vc_removal(1.5)
    clusterer.slope_removal(10 ** -3)
    features = ['min_rhp', 'vol_scaled', 'height_scaled', 'el_min_scaled']
    clusterer.preprocess_features(features)
    
    clusterer.clusterer = KMeans(n_clusters=10, random_state=0)
    clusterer.cluster()
    clusterer.plot_edzs()

def main():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.feature_data['fp_slope'] = (clusterer.feature_data['w_edep'] - clusterer.feature_data['w_edap']) / clusterer.feature_data['height']
    clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] <= 0, 'fp_slope'] = clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] > 0, 'fp_slope'].min()
    clusterer.vc_removal(1.5)
    clusterer.slope_removal(10 ** -3)
    # features = ['rhp_pre', 'rhp_post', 'el_edap_scaled', 'vol']
    # features = ['rhp_pre', 'rhp_post', 'height_scaled', 'height', 'el_edap_scaled', 'vol', 'min_rhp', 'cumulative_volume']
    # features = ['w_edap', 'w_edep', 'el_edap', 'fp_slope', 'valley_confinement']
    features = ['w_edep', 'el_edep', 'w_edap', 'el_edap', 'fp_slope', 'valley_confinement', 'vol', 'min_rhp', 'cumulative_volume', 'cumulative_height', 'height', 'height_scaled', 'slope_min_stop']
    features = ['ave_rhp', 'stdev_rhp']
    clusterer.preprocess_features(features)
    # clusterer.optimize_som()
    # clusterer.plot_features()
    # clusterer.clusterer = KMedoids(n_clusters=5, random_state=0)
    # clusterer.clusterer = KMeans(n_clusters=5, random_state=0)
    # clusterer.cluster()
    clusterer.calc_embedding(method='umap')
    # clusterer.vis_clusters()
    # clusterer.plot_cluster_hydraulics()
    clusterer.vis_celerity()

def exploratory():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.out_dir = os.path.join(clusterer.run_dict['analysis_directory'], 'exploratory_clustering')
    os.makedirs(clusterer.out_dir, exist_ok=True)

    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.feature_data['fp_slope'] = (clusterer.feature_data['w_edep'] - clusterer.feature_data['w_edap']) / clusterer.feature_data['height']
    clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] <= 0, 'fp_slope'] = clusterer.feature_data.loc[clusterer.feature_data['fp_slope'] > 0, 'fp_slope'].min()
    clusterer.vc_removal(1.5)
    clusterer.slope_removal(10 ** -3)
    features = ['w_edep', 'el_edep_scaled', 'w_edap', 'el_edap_scaled', 'fp_slope', 'valley_confinement', 'vol', 'min_rhp', 'el_min_scaled', 'cumulative_volume', 'height_scaled', 'slope_min_stop', 'slope_start_min', 'el_bathymetry_scaled', 'edz_count', 'DASqKm', 'slope']
    # features = ['ave_rhp', 'stdev_rhp', 'Ave_Rh', 'min_rhp', 'rh_bottom', 'rh_edap', 'rh_min', 'rh_edap', 'rhp_pre', 'rhp_post', 'DASqKm']
    # features = ['ave_rhp', 'stdev_rhp', 'min_rhp', 'vol', 'height_scaled', 'rhp_pre', 'rhp_post']
    clusterer.preprocess_features(features)
    clusterer.clusterer = KMeans(n_clusters=5, random_state=0, n_init='auto')
    clusterer.elbow_plot()
    clusterer.calc_embedding(method='tsne')
    clusterer.vis_celerity(detrended=True)
    clusterer.feature_data['embed_x'] = clusterer.embedding[:, 0]
    clusterer.feature_data['embed_y'] = clusterer.embedding[:, 1]
    clusterer.preprocess_features(['embed_x', 'embed_y'])
    clusterer.clusterer = GaussianMixture(n_components=4, random_state=0)
    # clusterer.clusterer = DBSCAN(eps=0.1, min_samples=5)
    clusterer.cluster()
    clusterer.vis_clusters()
    clusterer.feature_data['cluster'].to_csv(os.path.join(clusterer.out_dir, 'clustered_data.csv'))

    def find_representative_points(X, labels):
        unique_labels = np.unique(labels)
        representative_points = []

        for label in unique_labels:
            cluster_points = X[labels == label]
            distances = pairwise_distances(cluster_points)
            avg_distances = np.mean(distances, axis=1)
            central_index = np.argmin(avg_distances)
            central_point = cluster_points[central_index]
            central_index = np.where((X == central_point).all(axis=1))[0][0]
            representative_points.append(central_index)

        return np.array(representative_points)

    representative_points = find_representative_points(clusterer.X, clusterer.feature_data['cluster'].to_numpy())
    medoid_dict = {i: clusterer.feature_data.index[pt] for i, pt in enumerate(representative_points)}
    clusterer.cpal = {i: c for i, c in enumerate(['#fdc161','#c1bae0','#badeab', '#03bde3', '#fbabb7', '#fbf493'])}
    clusterer.plot_simple_hydraulics(medoid_dict)


def college_try():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    
    # Remove confined settings and high slopes
    clusterer.slope_removal(10 ** -3)
    clusterer.vc_removal(1.5)

    # preprocess features
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    clusterer.preprocess_features(features)
    # clusterer.plot_features()

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=5, random_state=0)
    # clusterer.elbow_plot()
    clusterer.cluster()
    clusterer.feature_data['cluster'] = clusterer.feature_data['cluster'].map(lambda x: chr(x + 65))

    b_reachs = clusterer.feature_data[clusterer.feature_data['cluster'] == 'B'].index
    sub_cluster(run_path, features, b_reachs, n_clusters=2)

    clusterer.plot_cluster_hydraulics()
    clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['DASqKm', 'regression_valley_confinement'])
    clusterer.plot_attenuation()


def sub_cluster(run_path, trans_dict, ids, rename_dict, n_clusters=4):
    clusterer = Clusterer(run_path)
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    clusterer.feature_data = clusterer.feature_data[clusterer.feature_data.index.isin(ids)]
    clusterer.preprocess_features_fromdict(trans_dict)

    clusterer.clusterer = KMedoids(n_clusters=n_clusters, random_state=0)
    clusterer.cluster()
    return clusterer


def paper():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    plotter = Clusterer(run_path)
    plotter.feature_data['cluster'] = np.nan
    clusterer = Clusterer(run_path)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    
    # Remove confined settings and high slopes
    non_attenuators = clusterer.feature_data[np.logical_or(clusterer.feature_data['slope'] > 10 ** -3, clusterer.feature_data['valley_confinement'] < 1.5)].index
    plotter.feature_data.loc[non_attenuators, 'cluster'] = 0
    clusterer.slope_removal(10 ** -3)
    clusterer.vc_removal(1.5)

    # preprocess features
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    renames = ['EDAP', 'Height', 'Valley Width', 'Valley Confinement', 'Size', 'Abruptness', 'Slope']
    rename_dict = {k: v for k, v in zip(features, renames)}
    rename_dict['DASqKm'] = 'Drainage Area (sqkm)'
    rename_dict['regression_valley_confinement'] = 'Valley Confinement (Regression)'
    rename_dict['streamorder'] = 'Stream Order'
    rename_dict['celerity_detrended'] = 'Shape Celerity (m^(2/3))'
    rename_dict['celerity'] = 'Celerity (m/s)'
    plotter.feature_data = plotter.feature_data.rename(columns=rename_dict)
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    features = renames

    trans_dict = clusterer.preprocess_features(features)
    print(clusterer.X.min(axis=0))
    medoid_dict = {0: None}

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=5, random_state=0)
    # clusterer.elbow_plot()
    clusterer.cluster()
    existing_clusters = np.nanmax(plotter.feature_data['cluster'].unique()) + 1
    clusterer.feature_data['cluster'] = clusterer.feature_data['cluster'] + existing_clusters
    for c in clusterer.feature_data['cluster'].unique():
        ids = clusterer.feature_data[clusterer.feature_data['cluster'] == c].index
        plotter.feature_data.loc[ids, 'cluster'] = c
    # log medoid indices
    for i, c in enumerate(clusterer.clusterer.medoid_indices_):
        i_adj = i + existing_clusters
        medoid_dict[i_adj] = clusterer.feature_data.index.to_list()[c]
    
    junk_drawer = 5
    junk_reaches = clusterer.feature_data[clusterer.feature_data['cluster'] == junk_drawer].index
    sub_clusterer = sub_cluster(run_path, trans_dict, junk_reaches, rename_dict, n_clusters=2)
    print(sub_clusterer.X.min(axis=0))
    # sub_clusterer.elbow_plot()
    existing_clusters = np.nanmax(plotter.feature_data['cluster'].unique()) + 1
    sub_clusterer.feature_data['cluster'] = sub_clusterer.feature_data['cluster'] + existing_clusters
    for c in sub_clusterer.feature_data['cluster'].unique():
        ids = sub_clusterer.feature_data[sub_clusterer.feature_data['cluster'] == c].index
        plotter.feature_data.loc[ids, 'cluster'] = c
    # log medoid indices
    for i, c in enumerate(sub_clusterer.clusterer.medoid_indices_):
        i_adj = i + existing_clusters
        medoid_dict[i_adj] = sub_clusterer.feature_data.index.to_list()[c]

    plotter.feature_data = plotter.feature_data[~plotter.feature_data['cluster'].isna()]
    plotter.preprocess_features_fromdict(trans_dict)
    print(plotter.X.min(axis=0))
    plotter.X[plotter.feature_data['cluster'] == 0] = np.nan
    naming_dict = {0: 'X',
                   1: 'D',
                   2: 'A',
                   3: 'C',
                   4: 'B',
                   5: 'NA',
                   6: 'E1',
                   7: 'E2'}
    plotter.feature_data['cluster'] = plotter.feature_data['cluster'].map(lambda x: naming_dict[int(x)])
    medoid_dict = {naming_dict[int(k)]: v for k, v in medoid_dict.items()}
    # plotter.plot_cluster_hydraulics()
    # plotter.plot_cluster_hydraulics_2()
    # plotter.plot_attenuation()
    # plotter.plot_celerity()

    plotter.plot_simple_hydraulics(medoid_dict)
    plotter.plot_feature_boxplots()
    plotter.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order'])
    plotter.plot_routing()

    plotter.feature_data[['cluster']].to_csv(os.path.join(plotter.out_dir, 'clustered_data.csv'))


    # Run an ANOVA on the cluster results for celerity_detrended
    # anova_subset = plotter.feature_data[(plotter.feature_data['cluster'] != 'A')]
    # y = anova_subset['celerity_detrended']
    # x = anova_subset['cluster']
    # x = x.map(lambda x: ord(x) - 65)
    # x = sm.add_constant(x)
    # model = sm.OLS(y, x)
    # result = model.fit()
    # print(result.summary())

def main5():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    plotter = Clusterer(run_path)
    plotter.feature_data['cluster'] = np.nan
    plotter.out_dir = os.path.join(plotter.run_dict['analysis_directory'], 'slope_adjust')
    clusterer = Clusterer(run_path)
    clusterer.out_dir = os.path.join(clusterer.run_dict['analysis_directory'], 'slope_adjust')
    os.makedirs(plotter.out_dir, exist_ok=True)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    
    # Remove confined settings and high slopes
    non_attenuators = clusterer.feature_data[np.logical_or(clusterer.feature_data['slope'] > 10 ** -2, clusterer.feature_data['valley_confinement'] < 1.5)].index
    plotter.feature_data.loc[non_attenuators, 'cluster'] = 0
    clusterer.slope_removal(10 ** -2)
    clusterer.vc_removal(1.5)

    # preprocess features
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    renames = ['EDAP', 'Height', 'Valley Width', 'Valley Confinement', 'Size', 'Abruptness', 'Slope']
    rename_dict = {k: v for k, v in zip(features, renames)}
    rename_dict['DASqKm'] = 'Drainage Area (sqkm)'
    rename_dict['regression_valley_confinement'] = 'Valley Confinement (Regression)'
    rename_dict['streamorder'] = 'Stream Order'
    rename_dict['celerity_detrended'] = 'Shape Celerity (m^(2/3))'
    rename_dict['celerity'] = 'Celerity (m/s)'
    plotter.feature_data = plotter.feature_data.rename(columns=rename_dict)
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    features = renames

    trans_dict = clusterer.preprocess_features(features)
    medoid_dict = {0: None}

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=5, random_state=0)
    clusterer.elbow_plot()
    clusterer.cluster()
    existing_clusters = np.nanmax(plotter.feature_data['cluster'].unique()) + 1
    clusterer.feature_data['cluster'] = clusterer.feature_data['cluster'] + existing_clusters
    for c in clusterer.feature_data['cluster'].unique():
        ids = clusterer.feature_data[clusterer.feature_data['cluster'] == c].index
        plotter.feature_data.loc[ids, 'cluster'] = c
    # log medoid indices
    for i, c in enumerate(clusterer.clusterer.medoid_indices_):
        i_adj = i + existing_clusters
        medoid_dict[i_adj] = clusterer.feature_data.index.to_list()[c]

    plotter.feature_data = plotter.feature_data[~plotter.feature_data['cluster'].isna()]
    plotter.preprocess_features_fromdict(trans_dict)
    plotter.X[plotter.feature_data['cluster'] == 0] = np.nan
    naming_dict = {0: 'X',
                   1: 'A',
                   2: 'B',
                   3: 'C',
                   4: 'D',
                   5: 'E',
                   6: 'F',
                   7: 'G'}
    plotter.feature_data['cluster'] = plotter.feature_data['cluster'].map(lambda x: naming_dict[int(x)])
    medoid_dict = {naming_dict[int(k)]: v for k, v in medoid_dict.items()}

    plotter.cpal['E'] = '#fa66ee'
    plotter.cpal['F'] = '#fbabb7'
    kristen_plot(plotter.feature_data['Drainage Area (sqkm)'].to_numpy(), plotter.feature_data['cluster'].to_numpy(), plotter.cpal, plotter.out_dir)
    plotter.plot_simple_hydraulics(medoid_dict)
    plotter.plot_feature_boxplots()
    plotter.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order'])
    plotter.plot_routing()

    plotter.feature_data[['cluster']].to_csv(os.path.join(plotter.out_dir, 'clustered_data.csv'))


    # Run an ANOVA on the cluster results for celerity_detrended
    # anova_subset = plotter.feature_data[(plotter.feature_data['cluster'] != 'A')]
    # y = anova_subset['celerity_detrended']
    # x = anova_subset['cluster']
    # x = x.map(lambda x: ord(x) - 65)
    # x = sm.add_constant(x)
    # model = sm.OLS(y, x)
    # result = model.fit()
    # print(result.summary())

def kristen_plot(da, clusters, cpal, out_dir):
    anr_y = (0.96 * ((da / 2.59) ** 0.30)) / 3.28
    bieger_y = (0.26 * (da ** 0.287))
    c_list = [cpal[c] for c in clusters]
    pd_version = pd.DataFrame({'bf': bieger_y, 'cluster': clusters})
    pd_version = pd.melt(pd_version, id_vars='cluster', value_vars='bf', var_name='Cluster', value_name='Bieger Bankfull Depth (m)')
    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, gridspec_kw={'height_ratios': [1, 2.5]}, sharex=True)
    sns.kdeplot(pd_version, x='Bieger Bankfull Depth (m)', hue='cluster', palette=cpal, ax=ax[0], clip=(0, bieger_y.max()), fill=True)
    ax[1].scatter(bieger_y, anr_y, c=c_list, s=3, alpha=0.7)
    max_pt = max(bieger_y.max(), anr_y.max())
    ax[1].plot([0, max_pt], [0, max_pt], color='k', linestyle='--', alpha=0.7)
    ax[1].set(xlabel='Bieger Bankfull Depth (m)', ylabel='VT ANR Bankfull Depth (m)')
    fig.savefig(os.path.join(out_dir, 'bankfull_comparison.png'), dpi=300)


def main6():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data['cluster'] = np.nan
    clusterer.out_dir = os.path.join(clusterer.run_dict['analysis_directory'], 'clusters_2')

    # Clean data and add floodplain slope feature
    split_1 = clusterer.feature_data[(clusterer.feature_data['valley_confinement'] < 1.5) | (clusterer.feature_data['slope'] > 3 * (10 ** -3))].copy()
    split_1['cluster'] = -1
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.slope_removal(3 * (10 ** -3))
    clusterer.vc_removal(1.5)
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    trans_dict = clusterer.preprocess_features(features)
    print(f'n={clusterer.X.shape[0]}')

    # EDA
    # fig, ax = multi_elbow(clusterer.X)
    # fig.savefig(os.path.join(clusterer.out_dir, 'multi_elbow_plot.png'), dpi=300)
    # pca = PCA(n_components=len(features))
    # pca.fit(clusterer.X)
    # fig, ax = plt.subplots()
    # ax.bar(range(1, len(features) + 1), pca.explained_variance_ratio_)
    # ax2 = ax.twinx()
    # ax2.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    # ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    # ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    # fig.savefig(os.path.join(clusterer.out_dir, 'pca_plot.png'), dpi=300)

    # Dimensionality reduction check
    clusterer.calc_embedding(method='pca')
    # clusterer.vis_celerity(detrended=True, plot_name='pca_celerity_map.png')

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=4, random_state=0)
    clusterer.cluster()
    medoid_dict = {-1: None}
    for i, c in enumerate(clusterer.clusterer.medoid_indices_):
        medoid_dict[i] = clusterer.feature_data.index.to_list()[c]
    
    # Sub-cluster
    sub_clusterer = Clusterer(run_path)
    junk_drawer = 1
    junk_reaches = (clusterer.feature_data['cluster'] == junk_drawer)
    sub_clusterer.X = clusterer.X[junk_reaches]
    junk_reaches = clusterer.feature_data[junk_reaches].index
    sub_clusterer.feature_data = clusterer.feature_data.loc[junk_reaches]

    # Chack cluster count for sub-reaches
    # fig, ax = multi_elbow(sub_clusterer.X)
    # fig.savefig(os.path.join(clusterer.out_dir, 'sub_multi_elbow_plot.png'), dpi=300)
    
    # sub cluster
    sub_clusterer.clusterer = KMedoids(n_clusters=3, random_state=0)
    sub_clusterer.cluster()
    c_offset = clusterer.feature_data['cluster'].max() + 1
    clusterer.feature_data.loc[junk_reaches, 'cluster'] = sub_clusterer.feature_data.loc[junk_reaches, 'cluster'] + c_offset
    del medoid_dict[junk_drawer]
    for i, c in enumerate(sub_clusterer.clusterer.medoid_indices_):
        medoid_dict[i + c_offset] = sub_clusterer.feature_data.index.to_list()[c]

    # clusterer.vis_clusters()

    # Merge data back
    clusterer.feature_data = pd.concat([clusterer.feature_data, split_1])
    clusterer.X = np.append(clusterer.X, np.ones((split_1.shape[0], clusterer.X.shape[1])) * np.nan, axis=0)
    naming_dict = {-1: 'X',
                   4: 'A1',
                   6: 'A2',
                   5: 'A3',
                   2: 'B',
                   0: 'C',
                   3: 'D'}
    clusterer.cpal = {i: c for i, c in zip(sorted(naming_dict.values()), ['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366'])}
    clusterer.feature_data['cluster'] = clusterer.feature_data['cluster'].map(lambda x: naming_dict[int(x)])
    medoid_dict = {naming_dict[int(k)]: v for k, v in medoid_dict.items()}

    # vis hydraulics
    # clusterer.plot_simple_hydraulics(medoid_dict)

    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    renames = ['EDAP', 'Height', 'Valley Width', 'Valley Confinement', 'Size', 'Abruptness', 'Slope']
    rename_dict = {k: v for k, v in zip(features, renames)}
    rename_dict['DASqKm'] = 'Drainage Area (sqkm)'
    rename_dict['regression_valley_confinement'] = 'Valley Confinement (Regression)'
    rename_dict['streamorder'] = 'Stream Order'
    rename_dict['celerity_detrended'] = 'Shape Celerity (m^(2/3))'
    rename_dict['celerity'] = 'Celerity (m/s)'
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    # clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order'])
    # clusterer.plot_routing()

    clusterer.feature_data[['cluster']].to_csv(os.path.join(clusterer.out_dir, 'clustered_data.csv'))


def ml_ai_eda():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data['cluster'] = np.nan
    clusterer.out_dir = os.path.join(clusterer.run_dict['analysis_directory'], 'ml_ai_meeting')

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.slope_removal(3 * (10 ** -3))
    clusterer.vc_removal(1.5)
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    trans_dict = clusterer.preprocess_features(features)
    print(f'n={clusterer.X.shape[0]}')

    # EDA
    # fig, ax = multi_elbow(clusterer.X)
    # fig.savefig(os.path.join(clusterer.out_dir, 'multi_elbow_plot.png'), dpi=300)
    # pca = PCA(n_components=len(features))
    # pca.fit(clusterer.X)
    # fig, ax = plt.subplots()
    # ax.bar(range(1, len(features) + 1), pca.explained_variance_ratio_)
    # ax2 = ax.twinx()
    # ax2.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    # ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    # ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    # fig.savefig(os.path.join(clusterer.out_dir, 'pca_plot.png'), dpi=300)

    # Dimensionality reduction
    clusterer.calc_embedding(method='pca')
    # clusterer.vis_celerity(detrended=True, plot_name='pca_celerity_map.png')
    # clusterer.calc_embedding(method='som')
    # clusterer.vis_celerity(detrended=True, plot_name='som_celerity_map.png')
    # clusterer.calc_embedding(method='umap')
    # clusterer.vis_celerity(detrended=True, plot_name='umap_celerity_map.png')
    # clusterer.calc_embedding(method='tsne')
    # clusterer.vis_celerity(detrended=True, plot_name='tsne_celerity_map.png')

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=4, random_state=0)
    clusterer.cluster()
    medoid_dict = {0: None}
    for i, c in enumerate(clusterer.clusterer.medoid_indices_):
        medoid_dict[i] = clusterer.feature_data.index.to_list()[c]
    clusterer.vis_clusters()

    # vis hydraulics
    clusterer.cpal = {i: c for i, c in enumerate(['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF'])}
    clusterer.plot_simple_hydraulics(medoid_dict)

    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    renames = ['EDAP', 'Height', 'Valley Width', 'Valley Confinement', 'Size', 'Abruptness', 'Slope']
    rename_dict = {k: v for k, v in zip(features, renames)}
    rename_dict['DASqKm'] = 'Drainage Area (sqkm)'
    rename_dict['regression_valley_confinement'] = 'Valley Confinement (Regression)'
    rename_dict['streamorder'] = 'Stream Order'
    rename_dict['celerity_detrended'] = 'Shape Celerity (m^(2/3))'
    rename_dict['celerity'] = 'Celerity (m/s)'
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order'])
    clusterer.plot_routing()

def multi_elbow(data):
    ch_scores = {'kmeans': list(), 'spectral': list(), 'agglomerative': list(), 'kmedoids': list(), 'gmm': list()}
    sil_scores = {'kmeans': list(), 'spectral': list(), 'agglomerative': list(), 'kmedoids': list(), 'gmm': list()}
    cluster_counts = list(range(2, 20))
    for i in cluster_counts:
        kmeans = KMeans(n_clusters=i, n_init='auto', random_state=0)
        kmeans.fit(data)
        ch_scores['kmeans'].append(calinski_harabasz_score(data, kmeans.labels_))
        sil_scores['kmeans'].append(silhouette_score(data, kmeans.labels_))

        spectral = SpectralClustering(n_clusters=i, random_state=0)
        spectral.fit(data)
        ch_scores['spectral'].append(calinski_harabasz_score(data, spectral.labels_))
        sil_scores['spectral'].append(silhouette_score(data, spectral.labels_))

        agglomerative = AgglomerativeClustering(n_clusters=i)
        agglomerative.fit(data)
        ch_scores['agglomerative'].append(calinski_harabasz_score(data, agglomerative.labels_))
        sil_scores['agglomerative'].append(silhouette_score(data, agglomerative.labels_))

        kmedoids = KMedoids(n_clusters=i, random_state=0)
        kmedoids.fit(data)
        ch_scores['kmedoids'].append(calinski_harabasz_score(data, kmedoids.labels_))
        sil_scores['kmedoids'].append(silhouette_score(data, kmedoids.labels_))

        gmm = GaussianMixture(n_components=i, random_state=0)
        gmm.fit(data)
        ch_scores['gmm'].append(calinski_harabasz_score(data, gmm.predict(data)))
        sil_scores['gmm'].append(silhouette_score(data, gmm.predict(data)))

    fig, ax = plt.subplots(nrows=2)
    for i in ch_scores:
        ax[0].plot(cluster_counts, ch_scores[i], label=i, alpha=0.5)
        ax[1].plot(cluster_counts, sil_scores[i], alpha=0.5)

    ax[1].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('Calinski Harabasz Score')
    ax[1].set_ylabel('Silhouette Score')
    for i in cluster_counts:
        ax[0].axvline(i, color='k', linewidth=0.3, alpha=0.2)
        ax[1].axvline(i, color='k', linewidth=0.3, alpha=0.2)
    ax[0].set_xticks(cluster_counts)
    ax[1].set_xticks(cluster_counts)
    # add legend on figure margin above both figures
    ax[0].legend(loc='center right', fontsize=9)
    fig.tight_layout()
    return fig, ax

def hydrofabric():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/6/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data['cluster'] = np.nan
    clusterer.out_dir = os.path.join(clusterer.run_dict['analysis_directory'], 'comparison_5_1')
    os.makedirs(clusterer.out_dir, exist_ok=True)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.slope_removal(3 * (10 ** -3))
    clusterer.vc_removal(1.5)
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    trans_dict = clusterer.preprocess_features(features)
    print(f'n={clusterer.X.shape[0]}')

    # # EDA
    # fig, ax = multi_elbow(clusterer.X)
    # fig.savefig(os.path.join(clusterer.out_dir, 'multi_elbow_plot.png'), dpi=300)
    # pca = PCA(n_components=len(features))
    # pca.fit(clusterer.X)
    # fig, ax = plt.subplots()
    # ax.bar(range(1, len(features) + 1), pca.explained_variance_ratio_)
    # ax2 = ax.twinx()
    # ax2.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    # ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    # ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    # fig.savefig(os.path.join(clusterer.out_dir, 'pca_plot.png'), dpi=300)

    # Dimensionality reduction
    clusterer.calc_embedding(method='pca')
    # clusterer.vis_celerity(detrended=True, plot_name='pca_celerity_map.png')
    # clusterer.calc_embedding(method='som')
    # clusterer.vis_celerity(detrended=True, plot_name='som_celerity_map.png')
    # clusterer.calc_embedding(method='umap')
    # clusterer.vis_celerity(detrended=True, plot_name='umap_celerity_map.png')
    # clusterer.calc_embedding(method='tsne')
    # clusterer.vis_celerity(detrended=True, plot_name='tsne_celerity_map.png')

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=5, random_state=0)
    clusterer.cluster()
    medoid_dict = {0: None}
    for i, c in enumerate(clusterer.clusterer.medoid_indices_):
        medoid_dict[i] = clusterer.feature_data.index.to_list()[c]
    clusterer.vis_clusters()

    # vis hydraulics
    clusterer.cpal = {i: c for i, c in enumerate(['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF'])}
    clusterer.plot_simple_hydraulics(medoid_dict)

    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    renames = ['EDAP', 'Height', 'Valley Width', 'Valley Confinement', 'Size', 'Abruptness', 'Slope']
    rename_dict = {k: v for k, v in zip(features, renames)}
    rename_dict['DASqKm'] = 'Drainage Area (sqkm)'
    rename_dict['regression_valley_confinement'] = 'Valley Confinement (Regression)'
    rename_dict['streamorder'] = 'Stream Order'
    rename_dict['celerity_detrended'] = 'Shape Celerity (m^(2/3))'
    rename_dict['celerity'] = 'Celerity (m/s)'
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order'])
    # clusterer.plot_routing()

def main7():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/7/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data['cluster'] = np.nan
    os.makedirs(clusterer.out_dir, exist_ok=True)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.wbody_removal()
    clusterer.slope_removal(3 * (10 ** -3))
    clusterer.vc_removal(1.5)
    features = ['el_edap_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'vol', 'min_rhp', 'slope']
    features = ['el_edap_scaled', 'el_edep_scaled', 'vol', 'min_rhp', 'min_loc_ratio', 'cumulative_volume', 'rhp_pre', 'rhp_post']
    features = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'w_edap', 'valley_confinement', 'min_rhp', 'vol']
    trans_dict = clusterer.preprocess_features(features, norm_type='standard')
    print(f'n={clusterer.X.shape[0]}')

    # EDA
    fig, ax = multi_elbow(clusterer.X)
    fig.savefig(os.path.join(clusterer.out_dir, 'multi_elbow_plot.png'), dpi=300)
    pca = PCA(n_components=len(features))
    pca.fit(clusterer.X)
    fig, ax = plt.subplots()
    ax.bar(range(1, len(features) + 1), pca.explained_variance_ratio_)
    ax2 = ax.twinx()
    ax2.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    fig.savefig(os.path.join(clusterer.out_dir, 'pca_plot.png'), dpi=300)

    # Dimensionality reduction
    clusterer.calc_embedding(method='tsne')

    # Cluster
    clusterer.clusterer = KMeans(n_clusters=4, random_state=0)
    clusterer.cluster()
    clusterer.vis_clusters()

    # vis hydraulics
    clusterer.cpal = {i: c for i, c in enumerate(['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF'])}
    clusterer.plot_simple_hydraulics(clusterer.medoid_dict)

    rename_dict = {
        'el_edap_scaled': 'EDAP',
        'height_scaled': 'Height',
        'w_edep': 'Valley Width',
        'valley_confinement': 'Valley Confinement',
        'vol': 'Size',
        'min_rhp': 'Abruptness',
        'slope': 'Slope',
        'DASqKm': 'Drainage Area (sqkm)',
        'regression_valley_confinement': 'Valley Confinement (Regression)',
        'streamorder': 'Stream Order',
        'celerity_detrended': 'Shape Celerity (m^(2/3))',
        'celerity': 'Celerity (m/s)'
    }
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order', 'Slope'])
    clusterer.plot_routing()
    clusterer.feature_data[['cluster']].to_csv(os.path.join(clusterer.out_dir, 'clustered_data.csv'))

def main8():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/8/run_metadata.json'
    clusterer = Clusterer(run_path)
    clusterer.feature_data['cluster'] = np.nan
    os.makedirs(clusterer.out_dir, exist_ok=True)

    # Clean data and add floodplain slope feature
    clusterer.feature_data.loc[clusterer.feature_data['w_edep'] > 5000, 'w_edep'] = 5000
    clusterer.wbody_removal()
    clusterer.slope_removal(3 * (10 ** -3))
    clusterer.vc_removal(1.5)
    features = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']
    trans_dict = clusterer.preprocess_features(features, norm_type='standard')
    print(f'n={clusterer.X.shape[0]}')

    # # EDA
    # fig, ax = multi_elbow(clusterer.X)
    # fig.savefig(os.path.join(clusterer.out_dir, 'multi_elbow_plot.png'), dpi=300)
    # pca = PCA(n_components=len(features))
    # pca.fit(clusterer.X)
    # fig, ax = plt.subplots()
    # ax.bar(range(1, len(features) + 1), pca.explained_variance_ratio_)
    # ax2 = ax.twinx()
    # ax2.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    # ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    # ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    # fig.savefig(os.path.join(clusterer.out_dir, 'pca_plot.png'), dpi=300)

    # # Dimensionality reduction
    # clusterer.calc_embedding(method='tsne')

    # Cluster
    clusterer.clusterer = KMedoids(n_clusters=8, random_state=0)
    clusterer.cluster()
    # clusterer.vis_clusters()

    # vis hydraulics
    clusterer.cpal = {i: c for i, c in enumerate(['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF'])}
    clusterer.plot_simple_hydraulics(clusterer.medoid_dict)

    rename_dict = {
        'el_edap_scaled': 'EDAP',
        'height_scaled': 'Height',
        'w_edep': 'Valley Width',
        'valley_confinement': 'Valley Confinement',
        'vol': 'Size',
        'min_rhp': 'Abruptness',
        'slope': 'Slope',
        'DASqKm': 'Drainage Area (sqkm)',
        'regression_valley_confinement': 'Valley Confinement (Regression)',
        'streamorder': 'Stream Order',
        'celerity_detrended': 'Shape Celerity (m^(2/3))',
        'celerity': 'Celerity (m/s)'
    }
    clusterer.feature_data = clusterer.feature_data.rename(columns=rename_dict)
    clusterer.plot_feature_boxplots()
    clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order', 'Slope'])
    clusterer.feature_data[['cluster']].to_csv(os.path.join(clusterer.out_dir, 'clustered_data.csv'))

if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # main4()
    # main5()
    # paper()
    # exploratory()
    # ml_ai_eda()
    # main6()
    # hydrofabric()
    # main7()
    main8()
