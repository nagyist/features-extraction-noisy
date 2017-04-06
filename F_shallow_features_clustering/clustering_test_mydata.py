"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as hcluster


print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
##############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

def clustering_hierarchical(X):
    thresh = 10
    clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
    return clusters

def clustering_agglomerative(X):
    ag = AgglomerativeClustering(n_clusters=3,compute_full_tree=True, ).fit(X)
    return ag.labels_


def clustering_gmm(X): # TOO SLOW!!! DOESN'T SCALE ON FEATURES SIZE!!
    gmm = GaussianMixture(verbose=True).fit(X)
    return gmm.labels_

def clustering_kmeans(X, nclusters=2):
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X)
    return kmeans.labels_


def clustering_spectral(X):
    spectral = SpectralClustering(n_clusters=3).fit(X)
    return spectral.labels_


def clustering_affinity_prop(X, **kwargs):
    ##############################################################################
    # Compute Affinity Propagation
    af = AffinityPropagation(preference=-20).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    return af.labels_


def clustering_mean_shift(X, **kwargs):
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.40)

    ms = MeanShift(bandwidth=bandwidth) #, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    import numpy as np
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    return labels




def clustering_dbscan(X): # NOT WORKING
    import  numpy as np
    # Compute DBSCAN
    db = DBSCAN(eps=0.40, min_samples=2, metric='cosine', algorithm='brute').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    return db.labels_


from config import cfg
from imdataset import ImageDataset
cfg.init()
dataset = ImageDataset().load_hdf5("shallow_extracted_features/shallow_feat_dbp3120_train_ds.h5")
dataset_sub = dataset.sub_dataset_with_label(1252)
# good examples: 1843, 123, 43, 422
# hard examples: 421, 843, 93, 927

# 3115 (monument) --> decent with kmeans 3 clusters
# 3118 (monument) --> best MS quantile=0.20
# 1843, 123, 43, 422 --> perfect with: db = DBSCAN(eps=900, min_samples=2, metric='cityblock').fit(X)

# impressive! db = DBSCAN(eps=0.35, min_samples=2, metric='cosine', algorithm='brute').fit(X)
# Perfect: 2910 (1 cluster with eps=0.40, 2 clusters with eps=0.35)
# 2908: good with 0.40
# Perfect: 1843, 423, 422, 43, 123, 2976, 290, 963
# Excellent! 3115 (monument)
# Good: 927, 3099, 1954, 1378, 1143
# Good/excellent, but create 2 good clusters: 843
# Decent: 421, 1984
# Decent if we remove only the cluster outleirs.. 3118 (monument)

X = dataset_sub.data
labels_true = dataset_sub.labels

#labels, cluster_center_indices = clustering_affinity_prop(X)
#labels = clustering_mean_shift(X)
#labels = clustering_kmeans(X, nclusters=3)
labels = clustering_dbscan(X)
#labels = clustering_agglomerative(X)
#labels = clustering_affinity_prop(X)
#labels = clustering_hierarchical(X)
#labels = clustering_gmm(X) # TOO SLOW!! DOESN'T SCALE!
#labels = clustering_spectral(X) # una merda..



##############################################################################
# Plot result
import matplotlib.pyplot as plt


plt.close('all')
plt.figure(1)
plt.clf()

for i, fname in enumerate(dataset_sub.fnames):
    dataset_class_name = dataset_sub.getLabelStr(i)
    cluster_label = labels[i]
    plt.title("C: " + str(cluster_label) + " - " + fname )
    im = plt.imread("dataset/dbp3120/{}/{}".format(dataset_class_name, fname))
    plt.imshow(im)


# # Plot result
# from itertools import cycle
#
# plt.close('all')
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

