import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


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
            c = self.cpal[i]
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
            c = self.cpal[i]
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
    rhp_axs = _plot_rhp(self, rhp_axs)

    w_axs = axs[1, :]
    w_axs = _plot_sec(self, w_axs)
    w_axs = _plot_sec_medoid(self,w_axs, med_reach_dict)

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


def multi_elbow(self):
    data = self.X
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
    fig.savefig(os.path.join(self.out_dir, 'multi_elbow_plot.png'), dpi=300)


def pca(self):
    pca = PCA(n_components=len(self.feature_cols))
    pca.fit(self.X)
    fig, ax = plt.subplots()
    ax.bar(range(1, len(self.feature_cols) + 1), pca.explained_variance_ratio_)
    ax2 = ax.twinx()
    ax2.plot(range(1, len(self.feature_cols) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
    ax.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Explained Variance Ratio')
    ax2.set(ylim=(0, 1.1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], xlabel='Principal Component', ylabel='Cumulative Explained Variance Ratio')
    fig.savefig(os.path.join(self.out_dir, 'pca_plot.png'), dpi=300)