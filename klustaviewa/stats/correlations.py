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
    # nclusters = Clu2.max() #max(Clu2)
    nclusters = len(spikes_in_clusters)

    # Default masks.
    if masks is None:
        masks = np.ones((nPoints, nDims), dtype=np.float32)
    
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

    LogP = np.zeros((nPoints, nclusters))

    stats = {}

    for c in spikes_in_clusters:
        # MyPoints = np.nonzero(Clu2==c)[0]
        MyPoints = spikes_in_clusters[c]
        # MyFet2 = Fet2[MyPoints, :]
        # now, take the modified features here
        # MyFet2 = y[MyPoints, :]
        MyFet2 = np.take(y, MyPoints, axis=0)
        # if len(MyPoints) > nDims:
        # LogProp = np.log(len(MyPoints) / float(nPoints)) # log of the proportion in cluster c
        Mean = np.mean(MyFet2, axis=0).reshape((1, -1))
        
        if len(MyPoints) <= 1:
            CovMat = 1e-3*np.eye(nDims)
            stats[c] = (Mean, CovMat, 1e3*np.eye(nDims), 
                (1e-3)**nDims, len(MyPoints))
            continue
        
        
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

def compute_correlations_approximation(features, clusters, masks,
        clusters_to_update=None, similarity_measure=None):
    """Compute the correlation matrix between every pair of clusters.
    
    Use an approximation of the original Klusters grouping assistant, with
    an integral instead of a sum (integral of the product of the Gaussian 
    densities).
    
    A dictionary pairs => value is returned.
    
    Compute all (i, *) and (i, *) for i in clusters_to_update
    
    """
    nPoints = features.shape[0] #size(Fet1, 1)
    nDims = features.shape[1] #size(Fet1, 2)
    c = np.unique(clusters)
    spikes_in_clusters = dict([(clu, np.nonzero(clusters == clu)[0]) for clu in c])
    nclusters = len(spikes_in_clusters)
    
    stats = compute_statistics(features, features, spikes_in_clusters, masks)
    
    clusterslist = sorted(stats.keys())
    
    # New matrix (clu0, clu1) => new value
    C = {}

    if clusters_to_update is None:
        clusters_to_update = clusterslist
    
    # Update the new matrix on the rows and diagonals of the clusters to
    # update.
    for ci in clusters_to_update:
        mui, Ci, Ciinv, logdeti, npointsi = stats[ci]
        for cj in clusterslist:
            muj, Cj, Cjinv, logdetj, npointsj = stats[cj]
            
            dmu = (muj - mui).reshape((-1, 1))
            
            if similarity_measure == 'kl':
                # KL divergence between the clusters
                pij = .5 * (np.trace(np.dot(Cjinv, Ci)) + np.dot(np.dot(dmu.T, Cjinv), dmu) - logdeti + logdetj - nDims)
                pji = .5 * (np.trace(np.dot(Ciinv, Cj)) + np.dot(np.dot(dmu.T, Ciinv), dmu) - logdetj + logdeti - nDims)
                
                C[ci, cj] = np.exp(pij)[0,0]
                C[cj, ci] = np.exp(pji)[0,0]
                
            else:
                # pij is the probability that mui belongs to Cj:
                #    $$p_{ij} = w_j * N(\mu_i | \mu_j; C_j)$$
                # where wj is the relative size of cluster j
                # pii is the probability that mui belongs to Ci
                pij = np.log(2*np.pi)*(-nDims/2.)+(-.5*logdetj)+(-.5) * np.dot(np.dot(dmu.T, Cjinv), dmu)
                pji = np.log(2*np.pi)*(-nDims/2.)+(-.5*logdeti)+(-.5) * np.dot(np.dot(dmu.T, Ciinv), dmu)
                
                # nPoints is the total number of spikes.
                wi = float(npointsi) / nPoints
                wj = float(npointsj) / nPoints
                
                C[ci, cj] = wj * np.exp(pij)[0,0]
                C[cj, ci] = wi * np.exp(pji)[0,0]
    
    return C
    
def get_similarity_matrix(dic):
    """Return a correlation matrix from a dictionary. Normalization happens
    here."""
    clu0, clu1 = zip(*dic.keys())
    clusters = sorted(set(clu0).union(set(clu1)))
    nclusters = len(clusters)
    clumax = max(clusters) + 1
    matrix = np.zeros((nclusters, nclusters))
    
    # Relative clusters: cluster absolute => cluster relative
    clusters_rel = np.zeros(clumax, dtype=np.int32)
    clusters_rel[clusters] = np.arange(nclusters)
    
    for (clu0, clu1), value in dic.iteritems():
        matrix[clusters_rel[clu0], clusters_rel[clu1]] = value
        
    return matrix
    
def normalize(matrix, direction='column'):
    
    if direction == 'row':
        s = matrix.sum(axis=1)
    else:
        s = matrix.sum(axis=0)
    
    # Non-null rows.
    indices = (s != 0)
    
    # Row normalization.
    if direction == 'row':
        matrix[indices, :] *= (1. / s[indices].reshape((-1, 1)))
    
    # Column normalization.
    else:
        matrix[:, indices] *= (1. / s[indices].reshape((1, -1)))
    
    return matrix
    

compute_correlations = compute_correlations_approximation


