from cluster_tools import FpClusterer
import sys

# Set up clusterer
run_path = sys.argv[1]
features = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']
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
n = 6
clusterer.cluster('A', 'kmedoids', n_clusters=n)

# Reformatting and cleaning
rename_dict = {
    'el_edap_scaled': 'EDZ Access Stage',
    'el_edep_scaled': 'EDZ Exit Stage',
    'height_scaled': 'EDZ Stage Range',
    'w_edep': 'EDZ Width',
    'valley_confinement': 'EDZ Relative Width',
    'vol': 'Diagnostic Size',
    'min_rhp': 'Max Lateral Expansion',
    'slope': 'Slope',
    'DASqKm': 'Drainage Area',
    'regression_valley_confinement': 'Valley Confinement',
    'streamorder': 'Stream Order',
    'celerity_detrended': 'Shape Celerity (m^(2/3))',
    'celerity': 'Celerity (m/s)'
}
clusterer.feature_cols = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']  # changing order for plotting
clusterer.features = clusterer.features.rename(columns=rename_dict)
clusterer.feature_cols = [rename_dict[col] for col in clusterer.feature_cols]
relabels = {1: 'A', 0: 'B', 2: 'C', 5: 'D', 4: 'E', 3: 'F', 6: 'G', 'W': 'W', 'X': 'X'}
relabels = {1: 'I', 0: 'II', 2: 'III', 5: 'IV', 4: 'V', 3: 'VI', 6: 'G', 'W': 'W', 'X': 'X'}
clusterer.clusters['cluster'] = clusterer.clusters['cluster'].map(relabels)
for k, v in relabels.items():
    if k in clusterer.medoid_dict:
        clusterer.medoid_dict[v] = clusterer.medoid_dict[k]
remove = clusterer.clusters['cluster'].isna()
clusterer.features = clusterer.features[~remove]
clusterer.trans_features = clusterer.trans_features[~remove]
clusterer.clusters = clusterer.clusters[~remove]
clusterer.colors = ["#8f00cc", "#cc0000", "#cc7000", "#cdbc00", "#07cc00", "#00cccc", '#2b54d8', '#979797']

magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
lables = ['50% AEP', '10% AEP', '2% AEP', '1% AEP']

# cms_label = 'Attenuation Per Km (cms)'
# value_cols = [f'{m}_Medium_cms_attenuation_per_km' for m in magnitudes]
# for c in value_cols:
#     clusterer.features[c] = 0
# value_cols = [f'{m}_Medium_pct_attenuation_per_km' for m in magnitudes]
# for c in value_cols:
#     clusterer.features[c] = 0


# Plotting
clusterer.plot_summary()
# clusterer.plot_feature_boxplots(transformed=False)
# clusterer.plot_boxplots_general(['Drainage Area', 'Valley Confinement', 'Slope'])
# clusterer.plot_routing()
clusterer.save_clusters()
clusterer.save_all_data()