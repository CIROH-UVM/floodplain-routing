import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA
import umap
from minisom import MiniSom
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import json
from scipy.ndimage import gaussian_filter1d
import colorsys


class ClusterCollection:

    def __init__(self):
        self.clusters = None
        self.features = None
        self.trans_features = None
        self.embedding = None
        self.feature_cols = None
        self.norm_type = None
        self.medoid_dict = dict()
        self.colors = ["#FF5733", "#FFBD33", "#FF3381", "#33FFC8", "#3364FF", "#FF3364", "#33FF57", "#33C8FF", "#33FFBD", "#64FF33", "#BD33FF", "#FFC833", "#FF33BD", "#C8FF33", "#57FF33", "#FF33C8"]


    def manual_cluster(self, mask, label):
        self.clusters.loc[mask, 'cluster'] = label
        self.update_medoid(label)

    def update_medoid(self, c, manual=False):
        mask = self.clusters['cluster'] == c
        if manual:
            subset = self.features[mask][self.feature_cols].values()
        else:
            subset = self.trans_features[mask]
        subset = np.nan_to_num(subset)
        dists = pairwise_distances(subset, subset)
        medoid = np.argmin(dists.sum(axis=0))
        subset = self.clusters.index[mask]
        medoid = subset[medoid]
        self.medoid_dict[c] = medoid

    def preprocess_features(self, feature_cols, target_cls, norm_type='min-max'):
        self.feature_cols = feature_cols
        self.norm_type = norm_type
        mask = self.clusters['cluster'] == target_cls
        X = self.features[mask][feature_cols].values

        # Remove skew
        init_floor = X.min(axis=0)
        X = X - init_floor
        skew = self.features[feature_cols].skew()
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
            X = (X - x_means) / x_stdevs
        elif norm_type == 'min-max':
            X = (X - x_min) / x_range

        self.trans_features[mask, :] = X

    def cluster(self, target_cls, method='kmeans', n_clusters=3):
        mask = self.clusters['cluster'] == target_cls
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        elif method == 'kmedoids':
            clusterer = KMedoids(n_clusters=n_clusters, random_state=0)
        elif method == 'spectral':
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=0)

        labels = clusterer.fit_predict(self.trans_features[mask])
        self.clusters.iloc[mask, 0] = labels
        for l in labels:
            self.update_medoid(l)

    def calc_embedding(self, target_cls, method='umap', sigma=2, lr=6, epochs=5000):
        print('Calculating Embeddings')
        mask = self.clusters['cluster'] == target_cls
        X = self.trans_features[mask]
        if method == 'umap':
            reducer = umap.UMAP()
            self.embedding = reducer.fit_transform(X)
        elif method == 'som':
            pca = PCA(n_components=2)
            pca.fit(X)
            rat = (pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]) + 1
            y_dim = (5 * np.sqrt(len(X))) / rat
            x_dim = y_dim * rat
            y_dim = int(y_dim)
            x_dim = int(x_dim)
            som = MiniSom(x_dim, y_dim, X.shape[1], sigma=sigma, learning_rate=lr, random_seed=0)
            som.pca_weights_init(X)
            som.train(X, epochs)
            self.embedding = np.array([som.winner(x) for x in X])
            # add jitter
            self.embedding = self.embedding + (np.random.random(self.embedding.shape) - 0.5)
        elif method == 'pca':
            pca = PCA(n_components=2)
            pca.fit(X)
            self.embedding = pca.transform(X)
        elif method == 'tsne':
            tsne = TSNE(n_components=2, perplexity=30, n_iter=5000)
            self.embedding = tsne.fit_transform(X)

    def multi_elbow(self, target_cls, max_bins=20):
        mask = self.clusters['cluster'] == target_cls
        data = self.trans_features[mask]
        inertia_scores = {'kmeans': list(), 'spectral': list(), 'agglomerative': list(), 'kmedoids': list(), 'gmm': list()}
        ch_scores = {'kmeans': list(), 'spectral': list(), 'agglomerative': list(), 'kmedoids': list(), 'gmm': list()}
        sil_scores = {'kmeans': list(), 'spectral': list(), 'agglomerative': list(), 'kmedoids': list(), 'gmm': list()}
        cluster_counts = list(range(2, max_bins))
        for i in cluster_counts:
            kmeans = KMeans(n_clusters=i, n_init='auto', random_state=0)
            kmeans.fit(data)
            inertia_scores['kmeans'].append(kmeans.inertia_)
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
            inertia_scores['kmedoids'].append(kmedoids.inertia_)
            ch_scores['kmedoids'].append(calinski_harabasz_score(data, kmedoids.labels_))
            sil_scores['kmedoids'].append(silhouette_score(data, kmedoids.labels_))

            gmm = GaussianMixture(n_components=i, random_state=0)
            gmm.fit(data)
            ch_scores['gmm'].append(calinski_harabasz_score(data, gmm.predict(data)))
            sil_scores['gmm'].append(silhouette_score(data, gmm.predict(data)))

        fig, ax = plt.subplots(nrows=3, figsize=(6, 9))
        for i in ch_scores:
            ax[0].plot(cluster_counts, ch_scores[i], label=i, alpha=0.5)
            ax[1].plot(cluster_counts, sil_scores[i], alpha=0.5)
            if i == 'kmeans' or i == 'kmedoids':
                ax[2].plot(cluster_counts, inertia_scores[i], alpha=0.5)
        for i in cluster_counts:
            ax[0].axvline(i, color='k', linewidth=0.3, alpha=0.2)
            ax[1].axvline(i, color='k', linewidth=0.3, alpha=0.2)
            ax[2].axvline(i, color='k', linewidth=0.3, alpha=0.2)

        ax[0].set(ylabel='Calinski Harabasz Score', xticks=cluster_counts)
        ax[1].set(ylabel='Silhouette Score', xlabel='Number of Clusters', xticks=cluster_counts)
        ax[2].set(ylabel='Inertia', xlabel='Number of Clusters', xticks=cluster_counts)

        ax[0].legend(loc='center right', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'multi_elbow_plot.png'), dpi=300)


class FpClusterer(ClusterCollection):

    def __init__(self, run_path, feature_cols):
        super().__init__()
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
        self.features = feature_data.set_index(run_dict['id_field'])

        el_data = pd.read_csv(el_path)
        self.el_data = el_data.dropna(axis=1)

        el_scaled_data = pd.read_csv(el_scaled_path)
        self.el_scaled_data = el_scaled_data.dropna(axis=1)

        width_data = pd.read_csv(width_path)
        self.width_data = width_data.dropna(axis=1)

        cel_data = pd.read_csv(celerity_path)
        self.cel_data = cel_data.dropna(axis=1)
        self.features['celerity'] = 0.0
        for c in cel_data.columns:
            if not c in self.features.index:
                continue
            elif self.features.loc[c, 'edz_count'] > 0:
                tmp_el = el_scaled_data[c].values
                edap = self.features.loc[c, 'el_edap_scaled']
                edap = np.argmax(tmp_el > edap)
                edep = self.features.loc[c, 'el_edep_scaled']
                edep = np.argmax(tmp_el > edep)
                if edap >= edep:
                    edep = edap + 1
                self.features.loc[c, 'celerity'] = cel_data[c].values[edap:edep].mean()  # average celerity for EDZ
            else:
                self.features.loc[c, 'celerity'] = cel_data[c].values[0:500].mean()
        self.features['celerity_detrended'] = np.log(self.features['celerity']) - np.log((self.features['slope'] ** 0.5) * (1 / 0.07))
        self.features['celerity_detrended'] = np.exp(self.features['celerity_detrended'])

        rh_prime_data = pd.read_csv(rh_prime_path)
        # Clean Rh prime
        rh_prime_data.iloc[-1] = rh_prime_data.iloc[-2]
        rh_prime_data[:] = gaussian_filter1d(rh_prime_data.T, 15).T
        rh_prime_data[rh_prime_data < -3] = -3
        self.rh_prime_data = rh_prime_data.dropna(axis=1)

        # Add attenuation
        # if os.path.exists(run_dict['muskingum_path']):
        #     magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
        #     durations = ['Short', 'Medium', 'Long']

        #     with open(r'source/regressions.json', 'r') as f:
        #         regressions = json.loads(f.read())
        #     regressions = regressions['peak_flowrate']
        #     for m in magnitudes:
        #         peak_estimate = ((self.features['DASqKm'].to_numpy() / 2.59) ** regressions[m][1]) * regressions[m][0] * (1 / 35.3147)
        #         for d in durations:
        #             self.features[f'{m}_{d}_cms_attenuation'] = self.features[f'{m}_{d}_pct_attenuation'] * peak_estimate
        #             total_lengths = (self.features[f'{m}_{d}_dx'] * self.features[f'{m}_{d}_subreaches']) / 1000
        #             self.features[f'{m}_{d}_cms_attenuation_per_km'] = self.features[f'{m}_{d}_cms_attenuation'] / total_lengths

        self.feature_cols = feature_cols
        self.clusters = pd.DataFrame(index=self.features.index, columns=['cluster'])
        self.trans_features = np.empty((len(self.features), len(feature_cols)))
        self.trans_features[:] = np.nan


    def _plot_rhp(self, ax):
        clusters = self.clusters['cluster'].unique()
        n = len(clusters)
        ord = sorted(clusters)
        y_space = np.linspace(0, 6, self.rh_prime_data.shape[0])

        for i in range(n):
            mask = self.clusters['cluster'] == ord[i]

            # trace each rh' curve within the label
            for r in self.features[mask].iterrows():
                reach = r[0]
                rh_prime = self.rh_prime_data[reach].values
                c = self.cpal[ord[i]]
                ax[i].plot(rh_prime, y_space, c=c, alpha=0.1)
            
            # Formatting and labelling
            ax[i].set(title=f'n={len(self.features[mask])}', xlabel=r"${R}_{h}$'", xlim=(-3, 1), yticks=[], facecolor='#f5f5f5')
        ax[0].set(ylabel='Stage (x bkf)', yticks=range(7))
        return ax
    
    def _plot_sec(self, ax):
        clusters = self.clusters['cluster'].unique()
        n = len(clusters)
        ord = sorted(clusters)
        stage_space = np.linspace(0, 6, self.width_data.shape[0])
        stage_space = np.append(stage_space[::-1], stage_space)

        max_w = 400
        new_ws = dict()
        for r in self.features.iterrows():
            length = r[1]['length']
            r = r[0]
            widths = self.width_data[r].values / length
            widths = np.append(-widths[::-1], widths)
            widths = widths / 2
            new_ws[r] = widths

       # plot sections
        for i in range(n):
            mask = self.clusters['cluster'] == ord[i]
            for r in self.features[mask].iterrows():
                r = r[0]
                widths = new_ws[r]
                widths = widths + max_w
                c = self.cpal[ord[i]]
                ax[i].plot(widths, stage_space, c=c, alpha=0.1)
            ax[i].set(xlim=(0, max_w * 2), xlabel='Top-width (m)', xticks=[0, max_w * 2], yticks=[], facecolor='#f5f5f5')
        return ax

    def _plot_sec_medoid(self, ax):
        clusters = self.clusters['cluster'].unique()
        n = len(clusters)
        ord = sorted(clusters)
        stage_space = np.linspace(0, 6, self.width_data.shape[0])
        stage_space = np.append(stage_space[::-1], stage_space)

        new_ws = dict()
        for r in self.features.iterrows():
            length = r[1]['length']
            r = r[0]
            widths = self.width_data[r].values / length
            widths = np.append(-widths[::-1], widths)
            widths = widths / 2
            new_ws[r] = widths

        # plot sections
        max_w = 400
        max_s = 4
        for i in range(n):
            r = self.medoid_dict[ord[i]]
            ax[i].set(xlim=(0, max_w * 2), xlabel='Top-width (m)', xticks=[0, max_w * 2], yticks=[], facecolor='#f5f5f5')
            if r is None:
                continue
            widths = new_ws[r]
            widths = widths + max_w
            ax[i].plot(widths, stage_space, c='k', alpha=1)
            ax[i].set_ylim(0, max_s)
        ax[0].set(ylabel='Stage (x bkf)', yticks=range(7), ylim=(0, max_s))
        return ax

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
        transformed_df = pd.DataFrame(self.trans_features, columns=self.feature_cols)
        transformed_df['cluster'] = self.clusters['cluster'].to_numpy()
        ord = sorted(transformed_df['cluster'].unique())
        for col in self.feature_cols:
            for c in ord:
                subset = self.features[self.clusters['cluster'] == c]
                print(f'{c} {col} median: {subset[col].median()}')

        cols = int(np.ceil(np.sqrt(len(self.feature_cols))))
        rows = int(len(self.feature_cols) / cols) + 1
        cols = 3
        rows = np.ceil(len(self.feature_cols) / cols).astype(int)
        
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(13, 9), sharey=True, sharex=True)
        if self.norm_type == 'standard':
            ylims = (-3.25, 3.25)
            yticks = np.arange(-3, 3, 1)
        else:
            ylims = (0, 1)
            yticks = np.arange(0, 1, 0.2)
        if len(self.feature_cols) % cols == 1:
            axs[-1, 0].remove()
            axs[-1, 2].remove()
        elif len(self.feature_cols) % cols == 2:
            axs[-1, 2].remove()

        ax_list = [ax for ax in axs.flat if ax.axes is not None]
        for i, ax in enumerate(ax_list):
            c = self.feature_cols[i]
            sns.boxplot(x='cluster', y=c, data=transformed_df, ax=ax, palette=self.cpal, order=ord, showfliers=False)
            ax.set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5', ylim=ylims, yticks=yticks)
        
        fig.tight_layout()
        # fig.savefig(os.path.join(self.out_dir, 'feature_boxplots.png'), dpi=300)
        fig.savefig(os.path.join(self.out_dir, 'feature_boxplots.pdf'), dpi=300)

    def plot_boxplots_general(self, col_list):
        cols = len(col_list)
        rows = 1
        ord = sorted(self.clusters['cluster'].unique())
        for col in col_list:
            for c in ord:
                subset = self.features[self.clusters['cluster'] == c]
                print(f'{c} {col} median: {subset[col].median()}')

        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(4.33 * cols, 2.66 * rows), sharex=True)
        tmp_merge = self.features.merge(self.clusters, left_index=True, right_index=True)
        i = 0
        for c in col_list:
            sns.boxplot(x='cluster', y=c, data=tmp_merge, ax=axs[i], palette=self.cpal, order=ord)
            if self.features[c].skew() > 1:
                axs[i].set_yscale('log')
            axs[i].set(xlabel=None, ylabel=None, title=c, facecolor='#f5f5f5')
            i += 1

        fig.tight_layout()
        # fig.savefig(os.path.join(self.out_dir, 'misc_boxplots.png'), dpi=300)
        fig.savefig(os.path.join(self.out_dir, 'misc_boxplots.pdf'), dpi=300)

    def plot_summary(self):
        clusters = self.clusters['cluster'].unique()
        n = len(clusters)
        ord = sorted(clusters)
        self.cpal = {k: v for k, v in zip(ord, self.colors[:n])}
        fig, axs = plt.subplots(ncols=n, nrows=2, figsize=(n*2, 7), gridspec_kw={'height_ratios': [5, 1]})
            
        rhp_axs = axs[0, :]
        rhp_axs = self._plot_rhp(rhp_axs)

        w_axs = axs[1, :]
        w_axs = self._plot_sec(w_axs)
        w_axs = self._plot_sec_medoid(w_axs)
    
        fig.tight_layout()
        # fig.savefig(os.path.join(self.out_dir, 'cluster_summary.png'), dpi=300)
        fig.savefig(os.path.join(self.out_dir, 'cluster_summary.pdf'), dpi=300)

    def plot_routing(self):
        ord = sorted(self.clusters['cluster'].unique())
        magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
        lables = ['50% AEP', '10% AEP', '2% AEP', '1% AEP']

        cms_label = 'Attenuation Per Km (cms)'
        value_cols = [f'{m}_Medium_cms_attenuation_per_km' for m in magnitudes]
        rename_dict = {i: j for i, j in zip(value_cols, lables)}
        tmp_merge = self.features.merge(self.clusters, left_index=True, right_index=True)
        cms_melt = pd.melt(tmp_merge, id_vars='cluster', value_vars=value_cols, var_name='Event', value_name=cms_label)
        cms_melt['Event'] = cms_melt['Event'].apply(lambda x: rename_dict[x])
        # cms_melt[cms_label] = cms_melt[cms_label].clip(lower=0)
        cms_melt[cms_label][cms_melt[cms_label] < 0] = np.nan

        pct_label = 'Attenuation Per Km (pct)'
        value_cols = [f'{m}_Medium_pct_attenuation_per_km' for m in magnitudes]
        rename_dict = {i: j for i, j in zip(value_cols, lables)}
        pct_melt = pd.melt(tmp_merge, id_vars='cluster', value_vars=value_cols, var_name='Event', value_name=pct_label)
        pct_melt['Event'] = pct_melt['Event'].apply(lambda x: rename_dict[x])
        # pct_melt[pct_label] = pct_melt[pct_label].clip(lower=0)
        pct_melt[pct_label][pct_melt[pct_label] < 0] = np.nan

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 9))
        sns.boxplot(x='Event', y=cms_label, hue='cluster', data=cms_melt, ax=axs[0, 0], palette=self.cpal, showfliers=False, hue_order=ord)
        b = axs[0, 0].legend()
        b.remove()
        sns.boxplot(x='Event', y=pct_label, hue='cluster', data=pct_melt, ax=axs[0, 1], palette=self.cpal, showfliers=False, hue_order=ord)
        b = axs[0, 1].legend()
        b.remove()
        sns.boxplot(tmp_merge, x='cluster', y='Celerity (m/s)', ax=axs[1, 0], palette=self.cpal, order=ord, showfliers=False)
        sns.boxplot(tmp_merge, x='cluster', y='Shape Celerity (m^(2/3))', ax=axs[1, 1], palette=self.cpal, order=ord, showfliers=False)
        for i in range(2):
            for j in range(2):
                axs[i, j].set_facecolor('#f5f5f5')

        fig.tight_layout()
        # fig.savefig(os.path.join(self.out_dir, 'routing_plot.png'), dpi=300)
        fig.savefig(os.path.join(self.out_dir, 'routing_plot.pdf'), dpi=300)

    def save_clusters(self):
        self.clusters.to_csv(os.path.join(self.out_dir, 'clustered_features.csv'))

    def save_all_data(self):
        out_df = self.features.copy()
        cols = [f + '_transformed' for f in self.feature_cols]
        trans_features = pd.DataFrame(self.trans_features, columns=cols, index=self.features.index)
        out_df = pd.concat([out_df, trans_features], axis=1) 
        out_df = pd.concat([out_df, self.clusters], axis=1)
        out_df.to_csv(os.path.join(self.out_dir, 'all_data.csv'))    

    def plot_clusters(self):
        fig, ax = plt.subplots()
        for c in self.clusters['cluster'].unique():
            mask = self.clusters['cluster'] == c
            ax.scatter(self.embedding[mask, 0], self.embedding[mask, 1], c=self.cpal[c], label=c, alpha=0.5)
        ax.legend()
        fig.savefig(os.path.join(self.out_dir, 'cluster_plot.png'), dpi=300)

    def make_colors(self, n):
        hue_range = np.array((0, (275 * (n / (n + 1)))))
        hue_range += 275  # start at blue
        hues = np.linspace(hue_range[0], hue_range[1], n) / 360.0
        saturations = np.repeat(0.75, n)
        lightnesses = np.repeat(0.5, n)
        self.colors = [colorsys.hls_to_rgb(h, l, s) for h, s, l in zip(hues, saturations, lightnesses)]
        self.colors.append('#3675D3')
        self.colors.append('#736A59')

    