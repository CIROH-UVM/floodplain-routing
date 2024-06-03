from cluster_tools import FpClusterer

# Set up clusterer
run_path = r'/netfiles/ciroh/floodplainsData/runs/8/run_metadata.json'
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
    'el_edap_scaled': 'EDAP Stage',
    'el_edep_scaled': 'EDEP Stage',
    'height_scaled': 'Height',
    'w_edep': 'EDEP Width',
    'w_edep_scaled': 'EDEP Width Scaled',
    'w_edap': 'EDAP Width',
    'valley_confinement': 'Unconfinedness',
    'vol': 'EDZ Size',
    'min_rhp': 'Min RHP',
    'slope': 'Slope',
    'DASqKm': 'Drainage Area (sqkm)',
    'regression_valley_confinement': 'Valley Confinement (Regression)',
    'streamorder': 'Stream Order',
    'celerity_detrended': 'Shape Celerity (m^(2/3))',
    'celerity': 'Celerity (m/s)'
}
clusterer.feature_cols = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']  # changing order for plotting
clusterer.features = clusterer.features.rename(columns=rename_dict)
clusterer.feature_cols = [rename_dict[col] for col in clusterer.feature_cols]
# relabels = {0: 'A', 1: 'B', 3: 'C', 4: 'D', 2: 'E', 5: 'F', 'W': 'W', 'X': 'X'}
# relabels = {0: 'A', 1: 'D', 3: 'F', 4: 'B', 2: 'E', 5: 'C', 'W': 'W', 'X': 'X'}  # w slope and standard scaling
relabels = {1: 'A', 0: 'B', 2: 'C', 5: 'D', 4: 'E', 3: 'F', 6: 'G', 'W': 'W', 'X': 'X'}  # w slope and robust scaling
clusterer.clusters['cluster'] = clusterer.clusters['cluster'].map(relabels)
for k, v in relabels.items():
    if k in clusterer.medoid_dict:
        clusterer.medoid_dict[v] = clusterer.medoid_dict[k]
remove = clusterer.clusters['cluster'].isna()
clusterer.features = clusterer.features[~remove]
clusterer.trans_features = clusterer.trans_features[~remove]
clusterer.clusters = clusterer.clusters[~remove]
clusterer.colors = ['#ff3366', '#ffc300', '#0fab0f', '#4bff87', '#01eaff', '#cc33ff', '#3366ff', '#858585']
clusterer.colors = ['#DF2A2D', '#FF531F', '#F59700', '#358B22', '#1ADBC8', '#D019F0', '#3675D3', '#736A59']
clusterer.colors = ["#df2a2d", "#f68624", "#ecd400", "#88d600", "#11caab", "#8401ff", "#ff33bb", '#3675D3', '#736A59']
# clusterer.colors = ['#DF2A2D', '#F59700', '#1ADBC8', '#D019F0', '#3675D3', '#736A59']
# clusterer.colors = ["#FF6F61", "#5E227F", "#7AC142", "#FFD166", "#00BFB2", "#F29F05", "#E74856", "#8734DB", "#FF5F76", '#3675D3', '#736A59']
# clusterer.colors = ["#FF6F61", "#5E227F", "#7AC142", "#FFD166", "#00BFB2", "#F29F05", "#E74856", '#3675D3', '#736A59']
clusterer.colors = ["#740fd9", "#e25fb8", "#d42313", "#f09722", "#c6bd18", "#30ce47", "#2adfd6", '#2b54d8', '#979797']
clusterer.colors = ["#8f00cc", "#cc0000", "#cc7000", "#cdbc00", "#07cc00", "#00cccc", '#2b54d8', '#979797']


# Plotting
clusterer.plot_summary()
clusterer.plot_feature_boxplots()
clusterer.plot_boxplots_general(['Drainage Area (sqkm)', 'Valley Confinement (Regression)', 'Stream Order', 'Slope', 'EDAP Width'])
clusterer.plot_routing()
clusterer.save_clusters()
clusterer.save_all_data()