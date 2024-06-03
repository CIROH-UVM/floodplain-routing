import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load regressions
with open('source/regressions.json') as in_file:
    REGRESSIONS = json.load(in_file)

def generate_boxplots(run_path):
    with open(run_path, 'r') as f:
        run_dict = json.loads(f.read())
    
    scaled_stage_path = os.path.join(run_dict['geometry_directory'], 'el_scaled.csv')
    rh_path = os.path.join(run_dict['geometry_directory'], 'rh.csv')
    vol_path = os.path.join(run_dict['geometry_directory'], 'vol.csv')
    meta_path = run_dict['reach_meta_path']

    scaled_stages = pd.read_csv(scaled_stage_path)
    rh = pd.read_csv(rh_path)
    vol = pd.read_csv(vol_path)
    meta = pd.read_csv(meta_path)

    reach_list = meta['ReachCode'].astype(int).astype(str).unique()
    n_list = [0.03, 0.05, 0.07, 0.09, 0.12]
    ri_list = list(REGRESSIONS['peak_flowrate'].keys())

    reaches = list()
    events = list()
    pis = list()
    ns = list()
    stages = list()
    

    counter = 1
    for reach in reach_list:
        print(f'reach {counter} / {len(reach_list)}')
        counter += 1
        tmp_meta = meta[meta['ReachCode'] == int(reach)]
        tmp_el_scaled = scaled_stages[reach]
        tmp_rh = rh[reach]
        tmp_a = vol[reach] / tmp_meta['length'].values[0]
        tmp_slope = tmp_meta['slope'].values[0]
        tmp_da = tmp_meta['TotDASqKm'].values[0] / 2.59

        for n in n_list:
            tmp_q = (1 / n) * tmp_a * (tmp_rh ** (2 / 3)) * (tmp_slope ** 0.5)
            ris = np.array([REGRESSIONS['peak_flowrate'][r][0] * (tmp_da ** REGRESSIONS['peak_flowrate'][r][1]) for r in ri_list]) / 35.3147
            modifiers = np.array([10 ** (1.656 * REGRESSIONS['ave_std_err'][r]) for r in ri_list])
            all_ris = np.array([*(ris / modifiers), *ris, *(ris * modifiers)])
            tmp_stages = np.interp(all_ris, tmp_q, tmp_el_scaled)

            reaches.extend(np.repeat(reach, len(all_ris)))
            events.extend(np.tile(ri_list, 3))
            pis.extend(np.repeat(['Low (5%)', 'Mean', 'High (95%)'], len(ri_list)))
            ns.extend(np.repeat(n, len(all_ris)))
            stages.extend(tmp_stages)
    out_df = pd.DataFrame({'Reach': reaches, 'Event': events, 'Confidence Interval': pis, "Manning's n": ns, 'Stage (X-bkf)': stages})
    sns.set_theme(style="ticks", palette="pastel")

    for i in ri_list:
        tmp_df = out_df[out_df['Event'] == i]
        ax = sns.boxplot(x="Confidence Interval", y="Stage (X-bkf)", hue="Manning's n", data=tmp_df)
        ax.axhline(1, ls='dashed', c='k', alpha=0.4, lw=1)
        ax.axhline(2, ls='dashed', c='k', alpha=0.4, lw=1)
        ax.set_ylim(-0.1, 5)
        ax.legend(title="Manning's n", loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=8)
        fig = ax.get_figure()
        fig.suptitle(i, x=0.1)
        fig.savefig(f'/netfiles/ciroh/floodplainsData/runs/4/working/mannings_calibration/{i}.png', dpi=300)
        plt.close(fig)
    



if __name__ == '__main__':
    run_path = r'/netfiles/ciroh/floodplainsData/runs/4/run_metadata.json'
    generate_boxplots(run_path)

