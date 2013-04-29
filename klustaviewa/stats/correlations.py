"""This module implements the computation of the correlation matrix between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np

from tools import matrix_of_pairs


# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------
def compute_statistics(Fet1, Fet2, spikes_in_clusters, masks):
    """Return Gaussian statistics about each cluster."""
    
    nPoints = Fet1.shape[0] #size(Fet1, 1)
    nDims = Fet1.shape[1] #size(Fet1, 2)
    # nClusters = Clu2.max() #max(Clu2)
    nClusters = len(spikes_in_clusters)

    # precompute the mean and variances of the masked points for each feature
    # contains 1 when the corresponding point is masked
    masked = np.zeros_like(masks)
    masked[masks == 0] = 1
    nmasked = np.sum(masked, axis=0)
    nu = np.sum(Fet2 * masked, axis=0) / nmasked
    # Handle nmasked == 0.
    nu[np.isnan(nu)] = 0
    nu = nu.reshape((1, -1))
    sigma2 = np.sum(((Fet2 - nu) * masked) ** 2, axis=0) / nmasked
    sigma2 = sigma2.reshape((1, -1))
    # expected features
    y = Fet1 * masks + (1 - masks) * nu
    z = masks * Fet1**2 + (1 - masks) * (nu ** 2 + sigma2)
    eta = z - y ** 2

    LogP = np.zeros((nPoints, nClusters))

    stats = {}

    for c in spikes_in_clusters:
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        # MyFet2 = Fet2[MyPoints, :]
        # now, take the modified features here
        # MyFet2 = y[MyPoints, :]
        MyFet2 = np.take(y, MyPoints, axis=0)
        # if len(MyPoints) > nDims:
        LogProp = np.log(len(MyPoints) / float(nPoints)) # log of the proportion in cluster c
        Mean = np.mean(MyFet2, axis=0).reshape((1, -1))
        CovMat = np.cov(MyFet2, rowvar=0) # stats for cluster c
        
        # HACK: avoid instability issues, kind of works
        CovMat += np.diag(1e-3 * np.ones(nDims))
        
        # now, add the diagonal modification to the covariance matrix
        # the eta just for the current cluster
        etac = np.take(eta, MyPoints, axis=0)
        d = np.sum(etac, axis=0) / nmasked
        
        # Handle nmasked == 0
        d[np.isnan(d)] = 0    
        
        # add diagonal
        CovMat += np.diag(d)
        CovMatinv = np.linalg.inv(CovMat)
        LogDet = np.log(np.linalg.det(CovMat))
        
        stats[c] = (Mean, CovMat, CovMatinv, LogDet, len(MyPoints))

    return stats

def compute_correlations(features, clusters, masks,
        clusters_to_update=None):
    """Compute the correlation matrix between every pair of clusters.
    
    Use an approximation of the original Klusters grouping assistant, with
    an integral instead of a sum (integral of the product of the Gaussian 
    densities).
    
    A dictionary pairs => value is returned.
    
    Compute all (i, *) and (i, *) for i in clusters_to_update
    
    """
    nPoints = features.shape[0] #size(Fet1, 1)
    nDims = features.shape[1] #size(Fet1, 2)
    c = Counter(clusters)
    spikes_in_clusters = dict([(clu, np.nonzero(clusters == clu)[0]) for clu in sorted(c)])
    nClusters = len(spikes_in_clusters)
    clumax = max(spikes_in_clusters.keys()) + 1
    
    stats = compute_statistics(features, features, spikes_in_clusters, masks)
    
    clusterslist = sorted(stats.keys())
    matrix_product = np.zeros((clumax, clumax))

    if clusters_to_update is None:
        clusters_to_update = clusterslist
    
    for ci in clusters_to_update:
        mui, Ci, Ciinv, logdeti, npointsi = stats[ci]
        for cj in clusterslist:
            muj, Cj, Cjinv, logdetj, npointsj = stats[cj]
            dmu = (muj - mui).reshape((-1, 1))
            
            p = np.log(2*np.pi)*(-nDims/2.)+(-.5*np.log(np.linalg.det(Ci+Cj)))+(-.5)*np.dot(np.dot(dmu.T, np.linalg.inv(Ci+Cj)), dmu)
            alpha = float(npointsi) / nPoints
            matrix_product[ci, cj] = np.exp(p + np.log(alpha))
            # matrix_product[cj, ci] = matrix_product[ci, cj]
    
    # Normalize the correlation matrix.
    s = matrix_product.sum(axis=1)
    matrix_product[s == 0, 0] = 1e-9
    s = matrix_product.sum(axis=1)
    matrix_product *= (1. / s.reshape((-1, 1)))
            
    d = {(ci, cj): matrix_product[ci, cj]
        for ci in clusters_to_update for cj in clusterslist}
    d.update({(ci, cj): matrix_product[ci, cj]
        for ci in clusterslist for cj in clusters_to_update})
    return d
    
    
    