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
l1 = 0.00000001
l2 = 0.1
transformation = lambda x: (((x + l2) ** l1) - 1) / l1
transformation = lambda x: np.log(x)

def yeo_johnson(x, lambda_):
    negs = x < 0
    poss = x >= 0
    if lambda_ == 0:
        x[poss] = np.log(x[poss] + 1)
    else:
        x[poss] = (((x[poss] + 1) ** lambda_) - 1) / lambda_
    if lambda_ == 2:
        x[negs] = -np.log((-x[negs]) + 1)
    else:
        x[negs] = -((((-x[negs]) + 1) ** (2 - lambda_)) - 1) / (2 - lambda_)
    return x

# transformation = lambda x: yeo_johnson(x, -4.5)


magnitudes = ['Q2', 'Q10', 'Q50', 'Q100']
durations = ['Short', 'Medium', 'Long']
def summary_table():
    metric = 'cms_attenuation_per_km'
    m_log = list()
    d_log = list()
    min_log = list()
    med_log = list()
    mean_log = list()
    max_log = list()
    for m, d in it.product(magnitudes, durations):
        col = f'{m}_{d}_{metric}'
        vals = results[col].dropna()
        if 'pct' in metric:
            vals = vals.clip(0, 1) * 100
        else:
            vals = vals.clip(0)
        m_log.append(m)
        d_log.append(d)
        min_log.append(vals.min())
        med_log.append(vals.median())
        mean_log.append(vals.mean())
        max_log.append(vals.max())
    out_df = pd.DataFrame({'Magnitude': m_log, 'Duration': d_log, 'Min': min_log, 'Median': med_log, 'Mean': mean_log, 'Max': max_log})
    out_df.to_csv(os.path.join(run_dict['analysis_directory'], f'{metric}_summary.csv'), index=False)

def cdf_curves():
    metric = 'cms_attenuation_per_km'
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True)
    for i in range(4):
        m = magnitudes[i]
        for d in durations:
            col = f'{m}_{d}_{metric}'
            vals = results[col].dropna().sort_values()
            vals = vals + 0.1
            vals = ((vals ** 0.25) - 1) / 0.25
            axs[i].plot(vals, np.linspace(0, 1, len(vals)), label=d)
        axs[i].set_title(f'{m}')
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_cdfs.png'))

def kde_plots():
    metric = 'pct_attenuation_per_km'
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True)
    for i in range(4):
        m = magnitudes[i]
        for d in durations:
            col = f'{m}_{d}_{metric}'
            vals = results[col].dropna().sort_values()
            vals = transformation(vals)
            sns.kdeplot(vals, ax=axs[i], label=d)
        axs[i].set_title(f'{m}')
        # axs[i].set_xlim(-2, -1.5)
    fig.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_kde.png'))

def strip_plot():
    metric = 'pct_attenuation_per_km'
    cols = [f'{m}_{d}_{metric}' for m, d in it.product(magnitudes, durations)]
    melt = pd.melt(results[cols], var_name='event', value_name='att').dropna()
    melt['att'] = transformation(melt['att'])
    melt['magnitude'] = melt['event'].apply(lambda x: x.split('_')[0])
    melt['duration'] = melt['event'].apply(lambda x: x.split('_')[1])
    fig, ax = plt.subplots()
    sns.stripplot(data=melt, x='att', y='magnitude', hue='duration', ax=ax, dodge=True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_strip.png'))
            
def transformation_optimization():
    metric = 'cms_attenuation_per_km'
    
    for m, d in it.product(magnitudes, durations):
        nan_vals = results.index[results[f'{m}_{d}_mass_conserve'] < 0.9]
        results.loc[nan_vals, f'{m}_{d}_{metric}'] = np.nan
    cols = [f'{m}_{d}_{metric}' for m, d in it.product(magnitudes, durations)]
    tmp_df = pd.melt(results[cols], var_name='event', value_name='att').dropna(axis=0)
    print(f"l2 = {tmp_df['att'].min()}")
    tmp_df['att'] = tmp_df['att'] - tmp_df['att'].min()

    ks = list()
    ad = list()
    sh = list()
    n = list()
    param_range = np.linspace(-5, 5, 100)
    for l in param_range:
        if l == 0:
            continue
        trans_data = yeo_johnson(tmp_df['att'].to_numpy(), l)
        ks.append(kstest(trans_data, 'norm').statistic)
        ad.append(anderson(trans_data).statistic)
        sh.append(shapiro(trans_data).statistic)
        n.append(normaltest(trans_data).statistic)
    
    fig, axs = plt.subplots(nrows=4, sharex=True, figsize=(9, 4))
    axs[0].plot(param_range, ks)
    axs[0].set_title('Kolmogorov-Smirnov')
    axs[0].text(param_range.max(), max(ks), 'Higher is more normal', ha='right', va='top')
    axs[1].plot(param_range, ad)
    axs[1].set_title('Anderson-Darling')
    axs[1].text(param_range.max(), max(ad), 'Lower is more normal', ha='right', va='top')
    axs[2].plot(param_range, sh)
    axs[2].set_title('Shapiro-Wilk')
    axs[2].text(param_range.max(), max(sh), 'Higher is more normal', ha='right', va='top')
    axs[3].plot(param_range, n)
    axs[3].set_title('Normal Test')
    axs[3].text(param_range.max(), max(n), 'Lower is more normal', ha='right', va='top')
    for i in range(4):
        axs[i].set_xticks(np.arange(-5, 5, 0.5))
        axs[i].grid(axis='x', which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_transformation_optimization.png'))


def pareto():
    col = 'Q2_Medium_cms_attenuation_per_km'
    vals = results[col].dropna().clip(0)
    vals = vals.sort_values(ascending=False)
    cumvals = (vals.cumsum() / vals.sum()) * 100
    arg_80 = np.argmax(cumvals > 80)
    print(f'Pareto Principle Check: {round(arg_80 / len(vals), 2)}% of reaches account for 80% of cubic meters attenuated')


def linreg_1():
    metric = 'cms_attenuation_per_km'
    
    tmp_cols = [f'{m}_{d}_{metric}' for m, d in it.product(magnitudes, durations)]
    melt = pd.melt(results[tmp_cols], var_name='event', value_name='att', ignore_index=False)
    melt = melt.join(results[['DASqKm', 'slope']], how='left')
    melt['slope'] = np.log(melt['slope'])
    melt['DASqKm'] = np.log(melt['DASqKm'])

    melt = melt.dropna(axis=0)
    for d in durations:
        melt[d] = melt['event'].apply(lambda x: (x.split('_')[1] == d) * 1)
    for m in magnitudes:
        melt[m] = melt['event'].apply(lambda x: (x.split('_')[0] == m) * 1)
    y = melt['att']
    x = melt[['DASqKm', 'slope', 'Short', 'Medium', 'Long', 'Q2', 'Q10', 'Q50', 'Q100']]


    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2)
    est2 = est.fit()
    print(est2.summary())
    # save to csv
    # out_df = pd.DataFrame({'variable': ['int', 'DASqKm', 'slope', 'Short', 'Medium', 'Long', 'Q2', 'Q10', 'Q50', 'Q100'], 'Coefficient': est2.params, 'P-Value': est2.pvalues})
    # out_df.to_csv(os.path.join(run_dict['analysis_directory'], f'{metric}_linreg.csv'), index=False)
    
    # print(f'{m} Linear Regression')
    # print('R-Squared_adj:', est2.rsquared_adj)
    # print('P-Values:', est2.pvalues)
    # print('Coefficients:', est2.params)
    # print('='*50)

def linreg_test():
    metric = 'cms_attenuation_per_km'

    tmp_cols = [f'{m}_{d}_{metric}' for m, d in it.product(magnitudes, durations)]
    melt = pd.melt(results[tmp_cols], var_name='event', value_name='att', ignore_index=False)
    melt = melt.dropna(axis=0)
    melt = melt.join(results[['DASqKm', 'slope']], how='left')
    melt['slope'] = np.log10(melt['slope'])
    melt['DASqKm'] = np.log10(melt['DASqKm'])
    melt['att'] = melt['att']
    melt['att'] = melt['att'].clip(1e-4)
    melt['att'] = np.log10(melt['att'])
    y = melt['att']
    x = melt[['slope']]
    x = sm.add_constant(x)
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())

    # fig, axs = plt.subplots(nrows=2, sharey=True)
    # axs[0].scatter(melt['DASqKm'], y, s=5, alpha=0.2)
    # axs[1].scatter(melt['slope'], y, s=5, alpha=0.2)
    
    # x1_space = np.linspace(melt['DASqKm'].min(), melt['DASqKm'].max(), 100)
    # x1_mean = melt['DASqKm'].mean()
    # x2_space = np.linspace(melt['slope'].min(), melt['slope'].max(), 100)
    # x2_mean = melt['slope'].mean()
    # pred1 = (est2.params['DASqKm'] * x1_space) + (est2.params['slope'] * x2_mean)
    # pred2 = (est2.params['DASqKm'] * x1_mean) + (est2.params['slope'] * x2_space)
    # axs[0].plot(x1_space, pred1, c='r', lw=2)
    # axs[1].plot(x2_space, pred2, c='r', lw=2)

    # fig.tight_layout()
    # fig.savefig(os.path.join(run_dict['analysis_directory'], f'test_{metric}_linreg.png'))

    fig, ax = plt.subplots()
    predy = est2.predict(x)
    ax.scatter(y, predy, s=5, alpha=0.2)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], c='r', ls='--')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'test_{metric}_linreg.png'))



def linreg_2():
    metric = 'cms_attenuation_per_km'

    tmp_cols = [f'{m}_{d}_{metric}' for m, d in it.product(magnitudes, durations)]
    melt = pd.melt(results[tmp_cols], var_name='event', value_name='att', ignore_index=False)
    melt = melt.dropna(axis=0)
    melt = melt.join(results[['DASqKm', 'slope']], how='left')
    melt['slope'] = np.log10(melt['slope'])
    melt['DASqKm'] = np.log10(melt['DASqKm'])
    melt['att'] = melt['att']
    melt['att'] = melt['att'].clip(1e-4)
    melt['att'] = np.log10(melt['att'])
    reg_cols = ['DASqKm', 'slope']
    for m, d in it.product(magnitudes, durations):
        melt[f'{m}{d}'] = (melt['event'] == f'{m}_{d}_{metric}') * 1
        reg_cols.append(f'{m}{d}')
    y = melt['att']
    x = melt[reg_cols]
    # x = sm.add_constant(x)
    est = sm.OLS(y, x)
    est2 = est.fit()
    print(est2.summary())
    # save to csv
    out_dict = est2.params.to_dict()
    for k in out_dict:
        out_dict[k] = [out_dict[k], est2.pvalues[k]]
    out_df = pd.DataFrame().from_dict(out_dict, orient='index', columns=['Coefficient', 'P-Value'])
    out_df.to_csv(os.path.join(run_dict['analysis_directory'], f'{metric}_linreg.csv'), index=True)

def da_scaling():
    coeffs = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'cms_attenuation_per_km_linreg.csv'), index_col=0)
    colors = {'Short': '#eb4034', 'Medium': '#439be8', 'Long': '#43e877'}
    lss = {'Short': 'dotted', 'Medium': 'solid', 'Long': 'dashed'}

    metric = 'cms_attenuation_per_km'
    fig, axs = plt.subplots(ncols=4, nrows=2, sharey=True, figsize=(12, 6))

    da = results['DASqKm']
    lda = np.log10(da)
    ldamn = lda.median()
    lda25 = lda.quantile(0.25)
    lda75 = lda.quantile(0.75)
    ldaspace = np.linspace(lda.max(), lda.min(), 100)
    c_da = coeffs.loc['DASqKm', 'Coefficient']

    s = results['slope']
    ls = np.log10(s)
    lsmn = ls.median()
    ls25 = ls.quantile(0.25)
    ls75 = ls.quantile(0.75)
    lsspace = np.linspace(ls.max(), ls.min(), 100)
    c_slope = coeffs.loc['slope', 'Coefficient']

    legend_dict = dict()
    for i in range(4):
        m = magnitudes[i]
        for d in durations:
            col = f'{m}_{d}_{metric}'
            y = results[col].clip(1e-4).to_numpy()
            axs[0, i].scatter(da, y, s=6, alpha=0.1, fc=colors[d])
            axs[1, i].scatter(s, y, s=3, alpha=0.1, fc=colors[d])

            # Linear Regressions
            c_event = coeffs.loc[f'{m}{d}', 'Coefficient']
            # p_da_low = (c_da * ldaspace) + (c_slope * ls75) + c_event
            # p_da_high = (c_da * ldaspace) + (c_slope * ls25) + c_event
            p_da_mid = (c_da * ldaspace) + (c_slope * lsmn) + c_event

            # p_slope_low = (c_da * lda25) + (c_slope * lsspace) + c_event
            # p_slope_high = (c_da * lda75) + (c_slope * lsspace) + c_event
            p_slope_mid = (c_da * ldamn) + (c_slope * lsspace) + c_event


            da_line = axs[0, i].plot(10 ** ldaspace, 10 ** p_da_mid, c='k', lw=1, alpha=0.9, ls=lss[d])
            # axs[0, i].fill_between(10 ** ldaspace, 10 ** p_da_low, 10 ** p_da_high, color=colors[d], alpha=0.2)

            axs[1, i].plot(10 ** lsspace, 10 ** p_slope_mid, c='k', lw=1, alpha=0.9, ls=lss[d])
            # axs[1, i].fill_between(10 ** lsspace, 10 ** p_slope_low, 10 ** p_slope_high, color=colors[d], alpha=0.9)

            legend_dict[d] = da_line[0]
        if True:
            ys = 'log'
        else:
            ys = 'linear'
        axs[0, i].set(title=f'{m}', xlabel='DA (SqKm)', ylabel='Attenuation (cms)', xscale='log', yscale=ys)
        axs[1, i].set(xlabel='Slope (m/m)', ylabel='Attenuation (cms)', xscale='log', yscale=ys)
    axs[1, 0].legend(legend_dict.values(), legend_dict.keys(), loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_da_scaling.png'))

def mass_conserve_check():
    metric = 'cms_attenuation_per_km'
    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(12, 6), sharex=True, sharey=True)
    for i in range(4):
        for j in range(3):
            x_col = f'{magnitudes[i]}_{durations[j]}_{metric}'
            y_col = f'{magnitudes[i]}_{durations[j]}_mass_conserve'
            x = results[x_col]
            y = results[y_col]
            axs[j, i].scatter(x, y, s=5, alpha=0.2, c='k')
            axs[j, i].axhline(1.1, c='r', ls='--', alpha=0.2)
            axs[j, i].axhline(0.9, c='r', ls='--', alpha=0.2)
            axs[j, i].set_title(f'{magnitudes[i]} {durations[j]}')
            if j == 2:
                axs[j, i].set_xlabel(metric)
            if i == 0:
                axs[j, i].set_ylabel('Mass Conservation')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], f'{metric}_mass_conserve.png'))
            

def slope_da_plot():
    coeffs = pd.read_csv(os.path.join(run_dict['analysis_directory'], 'cms_attenuation_per_km_linreg.csv'), index_col=0)
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True, sharey=True)
    x = np.log10(results['slope'])
    y = np.log10(results['DASqKm'])
    x_space = np.linspace(x.min(), x.max(), 100)
    y_space = np.linspace(y.min(), y.max(), 100)
    grid = np.meshgrid(x_space, y_space)
    min_att = np.inf
    max_att = -np.inf
    for i in range(4):
        for j in range(3):
            c1 = coeffs.loc[f'{magnitudes[i]}{durations[j]}DA', 'Coefficient']
            c2 = coeffs.loc[f'{magnitudes[i]}{durations[j]}slope', 'Coefficient']
            pred = (c1 * grid[1]) + (c2 * grid[0]) + coeffs.loc['const', 'Coefficient']
            min_att = min(min_att, pred.min())
            max_att = max(max_att, pred.max())
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(10 ** min_att, 10 ** max_att, 256))
    newcmp = ListedColormap(newcolors)
    for i in range(4):
        for j in range(3):
            m = magnitudes[i]
            d = durations[j]
            col = f'{m}_{d}_cms_attenuation_per_km'
            axs[j, i].scatter(x, y, s=2, alpha=0.2, c=results[col], cmap=newcmp)
            axs[j, i].set_title(f'{m} {d}')
            c1 = coeffs.loc[f'{magnitudes[i]}{durations[j]}DA', 'Coefficient']
            c2 = coeffs.loc[f'{magnitudes[i]}{durations[j]}slope', 'Coefficient']
            pred = (c1 * grid[1]) + (c2 * grid[0]) + coeffs.loc['const', 'Coefficient']
            axs[j, i].imshow(10 ** pred, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap=newcmp, alpha=0.5)
            if j == 2:
                axs[j, i].set_xlabel('Slope')
            if i == 0:
                axs[j, i].set_ylabel('DA')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dict['analysis_directory'], 'slope_da_plot.png'))

# transformation_optimization()
# cdf_curves()
# kde_plots()
# strip_plot()
# linreg_1()
# linreg_2()
# linreg_test()
da_scaling()
# mass_conserve_check()
# slope_da_plot()