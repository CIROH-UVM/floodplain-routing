import json
import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, SpectralClustering
from cluster_tools import Clusterer, ClusterCollection
from plot_tools import *

class FpClusterer(Clusterer):


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
            if not c in self.feature_data.index:
                continue
            elif self.feature_data.loc[c, 'edz_count'] > 0:
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

        # super().__init__(self.feature_data)

    def copy_data(self, other):
        other.el_data = self.el_data.copy()
        other.el_scaled_data = self.el_scaled_data.copy()
        other.width_data = self.width_data.copy()
        other.rh_prime_data = self.rh_prime_data.copy()
        other.cel_data = self.cel_data.copy()

def main():
    run_path = r'/netfiles/ciroh/floodplainsData/runs/7/run_metadata.json'
    primary = FpClusterer(run_path)
    features = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'w_edap', 'valley_confinement', 'min_rhp', 'vol']

    # Remove waterbodies
    wbody_mask = primary.feature_data['wbody'] == True
    wbodies = primary.feature_data[wbody_mask].copy()
    wbodies = Clusterer(wbodies)
    wbodies.X = np.empty((len(wbodies.feature_data), len(features)))
    wbodies.feature_data['cluster'] = 'W'
    wbodies.find_medoids()

    # Remove non-attenuators
    noatt_mask = (primary.feature_data['slope'] > (3 * (10 ** -3))) | (primary.feature_data['valley_confinement'] < 1.5)
    noatt_mask = noatt_mask & ~wbody_mask
    no_att = primary.feature_data[noatt_mask].copy()
    no_att = Clusterer(no_att)
    no_att.X = np.empty((len(no_att.feature_data), len(features)))
    no_att.feature_data['cluster'] = 'X'
    no_att.find_medoids()
    
    # Perform Initial Clustering
    l1_mask = ~(wbody_mask | noatt_mask)
    l1 = primary.feature_data[l1_mask].copy()
    l1 = Clusterer(l1)
    l1.out_dir = os.path.join(primary.run_dict['analysis_directory'], 'coarse_clustering')
    os.makedirs(l1.out_dir, exist_ok=True)
    primary.copy_data(l1)
    l1.preprocess_features(features, norm_type='standard')
    multi_elbow(l1)
    # l1.clusterer = KMeans(n_clusters=4, random_state=0)
    l1.clusterer = KMedoids(n_clusters=6, random_state=0)
    l1.cluster()
    l1.calc_embedding(method='pca')
    vis_clusters(l1)
    plot_simple_hydraulics(l1, l1.medoid_dict)
    relabels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
    l1.feature_data['cluster'] = l1.feature_data['cluster'].map(relabels)
    l1.medoid_dict = {relabels[k]: v for k, v in l1.medoid_dict.items()}

    # Consolidate clusters
    composite = ClusterCollection([l1, no_att, wbodies])
    composite.out_dir = os.path.join(primary.run_dict['analysis_directory'], 'composite_clustering')
    os.makedirs(composite.out_dir, exist_ok=True)
    primary.copy_data(composite)
    primary.feature_cols = features
    composite.compile()
    composite.cpal = {i: c for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F', 'W', 'X'], ['#FF5733', '#FFC300', '#219c21', '#3366FF', '#FF33EA', '#15e8d2', '#FF3366', '#CC33FF', '#33CCFF'])}
    plot_simple_hydraulics(composite, composite.medoid_dict)

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
    composite.feature_data = composite.feature_data.rename(columns=rename_dict)
    plot_feature_boxplots(composite)
    plot_boxplots_general(composite, ['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order', 'Slope'])
    


if __name__ == '__main__':
    main()