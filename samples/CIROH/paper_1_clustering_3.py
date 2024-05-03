from cluster_tools import FpClusterer
import json
import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Set up clusterer
run_path = r'/netfiles/ciroh/floodplainsData/runs/7/run_metadata.json'
features = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'w_edap', 'valley_confinement', 'min_rhp', 'vol']
clusterer = FpClusterer(run_path, features)

# Put waterbodies in a class
wbody_mask = clusterer.features['wbody'] == True
clusterer.manual_cluster(wbody_mask, 'W')

# Put non-attenuators in a class
noatt_mask = (clusterer.features['slope'] > (3 * (10 ** -3))) | (clusterer.features['valley_confinement'] < 1.5)
noatt_mask = noatt_mask & ~wbody_mask
clusterer.manual_cluster(noatt_mask, 'X')

# Cluster the rest
# Preprocess features
cluster_mask = ~(wbody_mask | noatt_mask)
clusterer.manual_cluster(cluster_mask, 'A')
clusterer.preprocess_features(features, 'A', norm_type='standard')
# clusterer.calc_embedding('A', 'pca')

# EDA
# clusterer.multi_elbow('A')

# Cluster
clusterer.cluster('A', 'kmedoids', n_clusters=6)

# Reformatting and cleaning
rename_dict = {
    'el_edap_scaled': 'EDAP',
    'el_edep_scaled': 'EDEP',
    'height_scaled': 'Height',
    'w_edep': 'Valley Width',
    'w_edap': 'Bankfull Width',
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
clusterer.features = clusterer.features.rename(columns=rename_dict)
clusterer.feature_cols = [rename_dict[col] for col in clusterer.feature_cols]
relabels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 'W': 'W', 'X': 'X'}
clusterer.clusters['cluster'] = clusterer.clusters['cluster'].map(relabels)
for k, v in relabels.items():
    clusterer.medoid_dict[v] = clusterer.medoid_dict[k]
remove = clusterer.clusters['cluster'].isna()
clusterer.features = clusterer.features[~remove]
clusterer.trans_features = clusterer.trans_features[~remove]
clusterer.clusters = clusterer.clusters[~remove]
clusterer.colors = ['#ff3366', '#ffc300', '#0fab0f', '#4bff87', '#01eaff', '#cc33ff', '#3366ff', '#858585']

# Plotting
clusterer.plot_summary()
clusterer.plot_feature_boxplots()
clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order', 'Slope'])
clusterer.plot_routing()
clusterer.save_clusters()