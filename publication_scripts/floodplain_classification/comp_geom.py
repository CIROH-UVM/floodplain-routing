import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def check_geom():
    d1 = r"/netfiles/ciroh/floodplainsData/runs/8/geometry"
    d2 = r"/netfiles/ciroh/floodplainsData/runs/9/geometry"

    comp_files = ['area', 'el', 'p', 'rh', 'rh_prime', 'vol']

    for f in comp_files:
        df1 = pd.read_csv(os.path.join(d1, f + '.csv'))
        df2 = pd.read_csv(os.path.join(d2, f + '.csv'))
        
        # compare by column
        print(f)
        comps = list()
        for c in df1.columns:
            try:
                close = np.allclose(df1[c].to_numpy(), df2[c].to_numpy(), rtol=1e-6)
                comps.append(close)
                if not close:
                    print(f'{c} not close')
            except:
                print(f'Error in column {c}')
        matches = np.sum(comps)
        print(f'{matches} matches out of {len(comps)}')

def check_mc():
    # d1 = r"/netfiles/ciroh/floodplainsData/runs/8/muskingum-cunge/mc_data.csv"
    d1 = r"/netfiles/ciroh/floodplainsData/runs/9/muskingum-cunge/mc_data.csv"
    d2 = r"/netfiles/ciroh/floodplainsData/runs/9/muskingum-cunge/mc_data_095.csv"
    # out_dir = r"/netfiles/ciroh/floodplainsData/runs/9/muskingum-cunge/comp_8to9"
    out_dir = r"/netfiles/ciroh/floodplainsData/runs/9/muskingum-cunge/debugging"

    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)

    combo = pd.merge(df1, df2, on='UVM_ID', suffixes=('_8', '_9'))

    magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
    duration = ['Short', 'Medium', 'Long']
    columns = ['pct_attenuation', 'pct_attenuation_per_km', 'cms_attenuation', 'cms_attenuation_per_km', 'mass_conserve', 'subreaches']
    for m in magnitudes:
        for d in duration:
            for c in columns:
                col = f'{m}_{d}_{c}'

                fig, ax = plt.subplots()
                ax.scatter(combo[col + '_8'], combo[col + '_9'])
                min_x = min(combo[col + '_8'].min(), combo[col + '_9'].min())
                max_x = max(combo[col + '_8'].max(), combo[col + '_9'].max())
                min_y = min(combo[col + '_8'].min(), combo[col + '_9'].min())
                max_y = max(combo[col + '_8'].max(), combo[col + '_9'].max())
                min_val = min(min_x, min_y)
                max_val = max(max_x, max_y)
                ax.plot([min_val, max_val], [min_val, max_val], ls='--', c='k')
                ax.set_title(col)
                ax.set_xlabel('8')
                ax.set_ylabel('9')
                valid = combo.dropna(subset=[col + '_8', col + '_9'])
                all_close = np.allclose(valid[col + '_8'].to_numpy(), valid[col + '_9'].to_numpy(), rtol=1e-2)
                ax.text(0.05, 0.05, f'All close: {all_close}', transform=ax.transAxes)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, col + '.png'))
                plt.close(fig)

def check_features():
    d1 = r"/netfiles/ciroh/floodplainsData/runs/8/analysis/data.csv"
    d2 = r"/netfiles/ciroh/floodplainsData/runs/9/analysis/data.csv"
    out_dir = r"/netfiles/ciroh/floodplainsData/runs/9/muskingum-cunge/comp_8to9"

    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)

    combo = pd.merge(df1, df2, on='UVM_ID', suffixes=('_8', '_9'))

    columns = ['el_edap_scaled', 'el_edep_scaled', 'height_scaled', 'w_edep', 'valley_confinement', 'min_rhp', 'vol']

    for col in columns:

        fig, ax = plt.subplots()
        ax.scatter(combo[col + '_8'], combo[col + '_9'])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='k')
        ax.set_title(col)
        ax.set_xlabel('8')
        ax.set_ylabel('9')
        valid = combo.dropna(subset=[col + '_8', col + '_9'])
        all_close = np.allclose(valid[col + '_8'].to_numpy(), valid[col + '_9'].to_numpy(), rtol=1e-2)
        ax.text(0.05, 0.05, f'All close: {all_close}', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, col + '.png'))
        plt.close(fig)

# check_geom()
check_mc()
# check_features()