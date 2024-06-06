import pandas as pd
import sys
import os
import json
import itertools as it

BREAK_STRING = '='*50

run_path = sys.argv[1]
with open(run_path) as f:
    run_dict = json.load(f)

# Get initial reach count and print some summary statistics
reach_meta = pd.read_csv(run_dict['reach_meta_path'])
reach_meta[run_dict['id_field']] = reach_meta[run_dict['id_field']].astype(int).astype(str)
reach_meta = reach_meta.set_index(run_dict['id_field'])
reach_meta['TotDASqKm'] = reach_meta['TotDASqKm'].clip(0, 2753)
initial_reaches = reach_meta.index.unique()
print(f'Initial reach count: {len(initial_reaches)}')
print(BREAK_STRING)

# DA stats
print(f'Min DA: {reach_meta["TotDASqKm"].min()}')
print(f'Max DA: {reach_meta["TotDASqKm"].max()}')
print(f'Mean DA: {reach_meta["TotDASqKm"].mean()}')
print(f'Median DA: {reach_meta["TotDASqKm"].median()}')
print(BREAK_STRING)

# Length stats
print(f'Min Length: {reach_meta["length"].min()}')
print(f'Max Length: {reach_meta["length"].max()}')
print(f'Mean Length: {reach_meta["length"].mean()}')
print(f'Median Length: {reach_meta["length"].median()}')
print(f'Total Length: {reach_meta["length"].sum() / 1000} km')
print(BREAK_STRING)

# Slope stats
print(f'Min Slope: {reach_meta["slope"].min()}')
print(f'Max Slope: {reach_meta["slope"].max()}')
print(f'Mean Slope: {reach_meta["slope"].mean()}')
print(f'Median Slope: {reach_meta["slope"].median()}')
print(BREAK_STRING)

# Load results data
results = pd.read_csv(run_dict['analysis_path'])
results[run_dict['id_field']] = results[run_dict['id_field']].astype(int).astype(str)
results = results.set_index(run_dict['id_field'])

# Analyze geometry errors
geom_errors = results.index[results['invalid_geometry'] == 1]
no_geo_err = results.index[results['invalid_geometry'] != 1]
print(f'Geometry error count: {len(geom_errors)}')
print(BREAK_STRING)

# Analyze MC errors
magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
durations = ['Short', 'Medium', 'Long']
for m, d in it.product(magnitudes, durations):
    col = f'{m}_{d}_pct_attenuation'
    error_count = results.loc[no_geo_err, col].isna().sum()
    print(f'{m}_{d} error count: {error_count}')
print(BREAK_STRING)

#  Assess volume conservation
all_cols = list()
for m, d in it.product(magnitudes, durations):
    col = f'{m}_{d}_mass_conserve'
    print(f'{m}_{d} mass conservation: {round(results[col].mean()*100, 2)}')
    all_cols.append(col)
print(BREAK_STRING)
all_vals = pd.melt(results[all_cols]).value
print(f'Min all volume conservation: {all_vals.min()}')
print(f'Max all volume conservation: {all_vals.max()}')
print(f'Mean all volume conservation: {all_vals.mean()}')
print(f'Median all volume conservation: {all_vals.median()}')
print(BREAK_STRING)


tmp_mc_results = results[['Q2_Medium_pct_attenuation', 'Q10_Medium_pct_attenuation', 'Q50_Medium_pct_attenuation', 'Q100_Medium_pct_attenuation']]
mc_errors = tmp_mc_results.index[tmp_mc_results.isna().any(axis=1)]
mc_errors_2 = tmp_mc_results.index[(tmp_mc_results < 0).any(axis=1)]
ave_att_error = tmp_mc_results.mean(axis=1)
print(f'mc geom error count: {len(mc_errors)} | percent: {round((len(mc_errors) / len(results)) * 100, 2)}')
print(f'negative attenuation error count: {len(mc_errors_2)} | percent: {round((len(mc_errors_2) / len(results)) * 100, 2)} | Average value of {round(ave_att_error.mean() * 100, 2)}% | median value of {round(ave_att_error.median() * 100, 2)}%')

# load clusters
clusters = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'clustering', 'clustered_features.csv'))
clusters[run_dict['id_field']] = clusters[run_dict['id_field']].astype(int).astype(str)
clusters = clusters.set_index(run_dict['id_field'])
cluster_labels = sorted(clusters['cluster'].unique())

for c in cluster_labels:
    cluster_reaches = clusters.loc[clusters['cluster'] == c]
    print(f'{c} reach count: {len(cluster_reaches)}')
    tmp_geom_errors = geom_errors.isin(cluster_reaches.index).sum()
    tmp_mc_errors = mc_errors.isin(cluster_reaches.index).sum()
    att_error = mc_errors_2.isin(cluster_reaches.index).sum()
    errs = ave_att_error.loc[cluster_reaches.index]
    errs = errs[errs < 0]
    print(f'geom error count: {tmp_geom_errors} | percent: {round((tmp_geom_errors / len(cluster_reaches)) * 100, 2)}')
    print(f'mc error count: {tmp_mc_errors} | percent: {round((tmp_mc_errors / len(cluster_reaches)) * 100, 2)}')
    print(f'negative attenuation error count: {att_error} | percent: {round((att_error / len(cluster_reaches)) * 100, 2)} | Average value of {round(errs.mean() * 100, 2)}% | median value of {round(errs.median() * 100, 2)}%')
    print(BREAK_STRING)

for m in magnitudes:
    d = 'Medium'
    col = f'{m}_{d}_pct_attenuation'
    negatives = (results[col] < 0).sum()
    print(f'{m}_{d} negative attenuation count: {negatives}  | percent: {round((negatives / len(results)) * 100, 2)}')
print(BREAK_STRING)


