# Clustering analyses

### Note that K-Means clustering is highly affected by outliers. Thus, outliers of over 2.5 std from the mean will be identified and excluded before running each clustering analysis.

# basic packages:
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from scipy import stats
import pingouin as pg
from sklearn.preprocessing import StandardScaler

# importing custom functions from this repository
from individual_differences_helper_functions import *

import warnings
warnings.simplefilter('ignore')

def cluster_data(cor_params, isOlder, isVividness):

    if isOlder:
        title_group = "Older"
    else:
        title_group = "Young"

    if isVividness:
        title_dv = "Vividness"
    else:
        title_dv = "Confidence"

    r_state=12
    prop_view = .8
    float_cols = ['gist_object','gist_person','gist_place','detail_object','detail_person','detail_place']

    myk = find_optimal_k(cor_params, r_state)
    # Run K-Means using myk clusters
    mykmeans = KMeans(n_clusters=myk, init="k-means++",
                      n_init=500,  #number of clustering attempts - returns lowest SSE
                      random_state=r_state).fit(cor_params)

    # fetch cluster centroids:
    centroids = pd.DataFrame(mykmeans.cluster_centers_, columns = cor_params.columns).reset_index()
    kmeans_labels = mykmeans.labels_
    cor_params = cor_params.astype('float')
    title = title_group + " adults clusters: " + title_dv + "-memory correlation patterns"
    plot_distance_network(cor_params, title, kmeans_labels, prop_view, myk, ran_seed=2)

    # save 
    distance_plot = "plots/distances_4cluster_"+ title_group + "_" +title_dv + ".pdf"
    plt.savefig(distance_plot, bbox_inches = 'tight')
    plt.show()

    print('N subjects per cluster =\n')
    print(pd.Series(kmeans_labels).value_counts().sort_index(),'\n')

    cluster_polar_plot(title, centroids, myk, -1.4, 1.8)

    # save 
    polar_plot = "plots/polar_4cluster_" + title_group + "_" +title_dv + ".pdf"
    plt.savefig(polar_plot, bbox_inches = 'tight')
    plt.show()

    cor_params['cluster'] = kmeans_labels

    wrapped_labels = [ label.replace('_', '\n') for label in float_cols ]

    plot_data = cor_params.reset_index().melt(value_vars=float_cols, var_name='Measure',
                                                 value_name='z', id_vars=['participant','cluster'])
    plot_data.z = plot_data.z.astype(float)

    plt.figure(figsize=(7,3.5))
    sns.barplot(data=plot_data, x='Measure', y='z',
                hue="cluster", palette="crest", alpha=.6,
                dodge=True)
    g = sns.stripplot(data=plot_data, x='Measure', y='z',
                      hue="cluster", palette="crest", dodge=True,
                      size=4, zorder=0, alpha=.8, jitter=.1)
    # Add legend
    handles, labels = g.get_legend_handles_labels()
    plt.legend(
               loc='upper right', bbox_to_anchor=(1.15, 1), 
               fontsize=10, title="Cluster", title_fontsize=12)

    plt.xlabel("")
    plt.xticks(list(range(len(wrapped_labels))), wrapped_labels,
               fontsize=16, wrap=True)
    plt.ylabel("Z", fontsize=20)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=22, y=1.05)
    plt.axhline(0,ls="-",color="black")  #chance
    plt.show()

    # run one-way anova within each measure and store stats as df (for bonferroni-correction)
    cor_params[float_cols] = cor_params[float_cols].astype('float')

    for f in range(len(float_cols)):
        stats = pg.anova(data=cor_params, dv=float_cols[f], between='cluster', effsize="n2", detailed=True)
        stats['measure'] = float_cols[f]
        if f == 0:
            all_stats = stats[stats.Source == 'cluster']
        else:
            all_stats = all_stats.append(stats[stats.Source == 'cluster'])
    return all_stats


def split_data(cor_params):
    older_df, younger_df = [x for _, x in cor_params.groupby(cor_params['age'] == 0)]
    older_df = older_df.drop(['age'], axis=1)
    younger_df = younger_df.drop(['age'], axis=1)

    float_cols = ['gist_object', 'gist_person', 'gist_place', 'detail_object', 'detail_person', 'detail_place']

    # standarise within each age group (no direct group comparisons will be made)
    scaler = StandardScaler()
    z_older_df = older_df.copy()
    z_older_df[float_cols] = scaler.fit_transform(older_df[float_cols])
    scaler = StandardScaler()
    z_younger_df = younger_df.copy()
    z_younger_df[float_cols] = scaler.fit_transform(younger_df[float_cols])

    # set outliers to nan
    z_older_df[z_older_df[float_cols].abs() > 2.5] = np.nan
    z_younger_df[z_younger_df[float_cols].abs() > 2.5] = np.nan

    print('\nOlder adult group memory performance outliers: ', z_older_df.loc[z_older_df.isnull().any(axis=1)].index.to_list())

    print('\nYounger adult group memory performance outliers: ', z_younger_df.loc[z_younger_df.isnull().any(axis=1)].index.to_list())

    # remove any subjects with a nan
    z_older_df.dropna(inplace=True)
    z_younger_df.dropna(inplace=True)
    return z_older_df, z_younger_df