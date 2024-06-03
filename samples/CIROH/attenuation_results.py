import json
import pandas as pd
import itertools as it
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import kstest, anderson, shapiro, normaltest
from sklearn import linear_model
import statsmodels.api as sm


run_path = r'/netfiles/ciroh/floodplainsData/runs/8/run_metadata.json'
with open(run_path) as f:
    run_dict = json.load(f)

# Load results data
results = pd.read_csv(run_dict['analysis_path'])
results['UVM_ID'] = results['UVM_ID'].astype(str)
results = results.set_index(run_dict['id_field'])
clusters = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'clustering', 'clustered_features.csv'))
clusters['UVM_ID'] = clusters['UVM_ID'].astype(str)
clusters = clusters.set_index(run_dict['id_field'])
subset = ['A', 'B', 'C', 'D', 'E', 'F']
results = results.join(clusters['cluster'], how='inner')
results = results[results['cluster'].isin(subset)]

magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
durations = ['Short', 'Medium', 'Long']


def event_trends(magnitude):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

    # Panel 1:  slope vs attenuation
    x = np.log10(results['slope'])
    y1 = results[f'{magnitude}_Medium_pct_attenuation_per_km']
    y2 = results[f'{magnitude}_Medium_cms_attenuation_per_km']
    axs[0, 0].scatter(x, y1)
    ax_00_twin = axs[0, 0].twinx()
    ax_00_twin.scatter(x, y2)

    