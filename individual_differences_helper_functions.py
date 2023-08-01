# Stores all functions for the individual differences analyses
# testing the relationship between vividness and memory attributes

# Rose Cooper - September 2021
# modified by Kasia Mojescik for purposes of the current study - March 2023

"""******************************
These analyses are based on the code provided by Cooper and Ritchey (2022), which can be found here: https://github.com/memobc paper-vividness-features.
 They have been extended to allow for an additional manipulation of an age group (both young and older adults have been tested) and an additional dependent variable - a confidence rating
 
 ******************************"""

#########################################

# imports: ----------------------------- #
# basic packages:
import pandas as pd
import numpy as np
import math

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import networkx as nx

# stats
from scipy import stats
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.spatial.distance import pdist, squareform
# -------------------------------------- #



# function for determining critical r across subjects
def critical_r(n, alpha, tail):
    df = n - 2
    if tail == 2:
        t = stats.t.ppf(alpha/2, df)
    elif tail == 1:
        t = stats.t.ppf(alpha, df)
    r = math.sqrt((t**2)/((t**2) + df))
    return(r)
# -------------------------------------- #



def create_memory_data(my_data):
    # creates a tidy dataframe with memory scores and one row per trial
    # same as function for individual experiment analyses

    # get scores for each feature:
    feature1 = my_data[['participant','age','event_id','q1_type','resp_gistmem1.corr','resp_detailmem1.corr']]
    feature1.columns=["participant","age","event","type","gist","detail"]

    feature2 = my_data[['participant','age','event_id','q2_type','resp_gistmem2.corr','resp_detailmem2.corr']]
    feature2.columns=["participant","age","event","type","gist","detail"]

    feature3 = my_data[['participant','age','event_id','q3_type','resp_gistmem3.corr','resp_detailmem3.corr']]
    feature3.columns=["participant","age","event","type","gist","detail"]

    # concatenate into long format
    feature_data = pd.concat([feature1,feature2,feature3], axis=0, ignore_index=True)

    # and format wider:
    feature_wide = feature_data.pivot_table(index=['participant','age','event'],columns='type').reset_index()


    # finally, merge with vividness scores:
    vividness_data = my_data[['participant','age','event_id','resp_ret_vividness.keys', 'resp_ret_confidence.keys']]
    vividness_data.columns=["participant","age","event","vividness","confidence"]
    vividness_data = vividness_data.merge(feature_wide, on=['participant','age','event'])

    # format column names so not tuples
    viv_columns = vividness_data.columns[5:11]
    vividness_data.columns = vividness_data.columns[0:5].tolist() + ['_'.join(i) for i in viv_columns]

    return vividness_data
# -------------------------------------- #



def rating_correlations(rating_data, features, rating):
    # runs within-subject correlations between vividness
    # and memory attributes
    # returns r

    cors = pd.DataFrame(index=rating_data['participant'].unique(),
                        columns=['age'] + features)
    for p in rating_data['participant'].unique():
        sub_data = rating_data[rating_data['participant'] == p]

        # correlate with rating
        sub_cors = sub_data[[rating] + features].corr(method="spearman").loc[rating,features].astype('float')
        cors_idx = (np.isnan(sub_cors)) | (np.round(np.abs(sub_cors),1) == 1)
        sub_cors[cors_idx] = 0
        cors.loc[p,features] = sub_cors

        # add group
        cors.loc[p,'age'] = sub_data.age.tolist()[0]

    return cors.reset_index()
# -------------------------------------- #



def fetch_sig_cors(cor_mat, n, sym=True, tail=1):
    # returns labels for significant correlations as an 
    # r x c matrix the same size as the input correlation matrix
    # * = p < .05 bonferroni-corrected
    # sym = True (symmetrical cor mat)
    # n = sample size
    # tail = 1 or 2 tailed correlation test

    # mark significant ones and plot:
    x_var = cor_mat.shape[0]
    y_var = cor_mat.shape[1]
    if sym:
        this_alpha = .05/((~np.isnan(cor_mat)).sum().sum() / 2)
    else:
        this_alpha = .05/((~np.isnan(cor_mat)).sum().sum())

    # critical r values (see top function)
    r_sig = critical_r(n, this_alpha, tail)

    # create mask
    p_mask = cor_mat >= r_sig
    d = {True: '*', False: ''}
    p_mask = np.asarray(p_mask.replace(d))

    # now create annotations (concatenate asterix with str cor value)
    labels = (np.asarray(["{0}\n{1:.2f}".format(sig, value)
                          for sig, value in zip(p_mask.flatten(),
                                                np.array(cor_mat).flatten())])
             ).reshape(x_var,y_var)

    return p_mask, labels, r_sig
# -------------------------------------- #



def plot_cor_heatmap(cor_mat, l = True, cmap="magma", title='', suptitle=''):
    # l = labels (see above function)

    #mask to remove diagonal (if symmetrical)
    mask = np.zeros_like(cor_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    if l == True:
        str_fmt = ".2f"
    else:
        str_fmt = "" #if input is pre-formatted labels

    # define figure size by dimensions:
    f_w = cor_mat.shape[1]*0.85
    f_h = cor_mat.shape[0]*0.75

    # colorbar range
    c_min = np.min([0.1,cor_mat[cor_mat != 1].min().min()])
    c_max = np.max([0.5,cor_mat[cor_mat != 1].max().max()])

    # heatmap
    plt.figure(figsize=(f_w,f_h), edgecolor="black")
    ax = sns.heatmap(cor_mat, mask=mask, annot=l,
                     cmap=cmap,vmin=c_min, vmax=c_max,
                     fmt=str_fmt, square=True)
    ax.set_facecolor('#e6e6e6')
    plt.xticks(fontsize=16, rotation=45, ha="right")
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=18, y=1.05)
    plt.suptitle(suptitle, fontsize=18, y=0.95, fontweight="bold")

    # custom colorbar axis
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=14)
    cax.set_ylabel('Spearman r', size=18, rotation=270, labelpad=25)

# -------------------------------------- #



def find_optimal_k(data, ran_state):
    # calculate the optimal number of clusters (k) to use for kmeans
    distortions = []
    K = range(1,11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, init="k-means++",
                            n_init=500,  #number of clustering attempts - returns lowest SSE
                            random_state=ran_state).fit(data)
        distortions.append(kmeanModel.inertia_)

    # use kneelocator to find the elbow
    myk = KneeLocator([*K], distortions, curve="convex", direction="decreasing").elbow
    print('Using',myk,'clusters for K-Means')

    # plot
    plt.figure(figsize=(5,2))
    plt.plot(K, distortions, 'o-', linewidth=3, markersize=10)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Inertia', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.title('Optimal k', fontsize=22, y=1.05)
    plt.axvline(x=myk, color="black", linewidth=2)
    plt.show()

    return myk
# -------------------------------------- #



def plot_distance_network(data, title, kmeans_labels, density, myk, ran_seed=100):
    # density = proportion of top edges (closest nodes) to keep

    # get distance matrix:
    dist_matrix = squareform(pdist(data.values, metric='euclidean'))

    # keep top X% of edges for visualization
    l_mat = np.tril(dist_matrix)
    dist_matrix[dist_matrix > np.quantile(l_mat[l_mat > 0],density)] = 0

    # color based on clusters
    this_cmap = cm.get_cmap("crest", myk)

    # create networkx graph object from thresholded matrix
    G = nx.from_numpy_matrix(dist_matrix)
    G.edges(data=True)
    pos = nx.spring_layout(G, seed=ran_seed)

    # draw network
    fig = plt.figure(figsize=(7,6))
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=100,
                           cmap=this_cmap, node_color=list(kmeans_labels)
                           )

    # add colorbar for communities
    sm = plt.cm.ScalarMappable(cmap=this_cmap, norm=plt.Normalize(vmin=.5, vmax=myk+.5))
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=[*range(1,myk+1)],
                      shrink=0.6)
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Cluster', fontsize=20, rotation=270, labelpad=25)

    plt.title(title, fontsize=22, y=1.05, x=0.55)
    plt.axis("off")

# -------------------------------------- #



def cluster_polar_plot(title, centroids,myk, lower_lim, upper_lim):
    # creates polar plot showing how clusters are driven by memory attributes

    # ------- PART 1: Create background
    # color based on clusters
    this_cmap = cm.get_cmap("crest", myk)

    # colormap as hex codes for seaborn
    this_hex = []
    for i in range(this_cmap.N):
        rgb = this_cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        this_hex.append(mpl.colors.rgb2hex(rgb))

    # number of variable
    categories=list(centroids.drop(columns='index').columns)
    N = len(categories)
    wrapped_labels = [label.replace('_', '\n') for label in categories]


    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(7,5))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], wrapped_labels,
               size=16, weight="bold")

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([-1, 0, 1], ["-1","0","1"], size=12)
    plt.ylim(lower_lim,upper_lim)

    plt.title(title,
              fontsize=22, x=0.55, y=1.15)


    # ------- PART 2: Add plots
    # Plot each group = each line of the data
    df = centroids.drop(columns='index')
    for c in df.index:
        values=df.loc[c,:].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=4, linestyle='solid',
                label= str(c+1), color=this_hex[c])
        ax.fill(angles, values, this_hex[c], alpha=0.2)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.7),
               fontsize=14, title="Cluster", title_fontsize=16)
    ax.spines['polar'].set_visible(False)

# -------------------------------------- #



