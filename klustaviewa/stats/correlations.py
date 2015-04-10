"""This module implements the computation of the correlation matrix between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np
from kwiklib.utils.logger import warn


# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------
class SimilarityMatrix(object):
    def __init__(self, features, masks):
        self.features = features
        nspikes, ndims = self.features.shape
        # Default masks.
        if masks is None:
            masks = np.ones((nspikes, ndims), dtype=np.float32)
        self.masks = masks
        self.unmask_threshold = 10
        self.clear_cache()
        self.compute_global_statistics()

    def clear_cache(self):
        self.stats = {}

    def compute_global_statistics(self):
        """Compute global Gaussian statistics from the features and masks."""

        nspikes, ndims = self.features.shape
        masks = self.masks
        features = self.features

        # precompute the mean and variances of the masked points for each
        # feature
        # this contains 1 when the corresponding point is masked
        masked = np.zeros_like(self.masks)
        masked[self.masks == 0] = 1
        nmasked = np.sum(masked, axis=0)
        nu = np.sum(self.features * masked, axis=0) / nmasked
        # Handle nmasked == 0.
        nu[np.isnan(nu)] = 0
        nu = nu.reshape((1, -1))
        sigma2 = np.sum(((self.features - nu) * masked) ** 2, axis=0) / nmasked
        sigma2[np.isnan(sigma2)] = 0
        # sigma2 = sigma2.reshape((1, -1))
        # WARNING: make sure what is inside diag is a 1D array, otherwise
        # it will take the diag of a 2D (1, n) matrix instead of generating a
        # (n, n) diagonal matrix...
        D = np.diag(sigma2.ravel())
        # expected features
        y = self.features * self.masks + (1 - self.masks) * nu
        z = self.masks * self.features**2 + (1 - self.masks) * (nu ** 2 + sigma2)
        eta = z - y ** 2

        self.y = y
        self.sigma2 = sigma2
        self.D = D
        self.eta = eta

    def compute_cluster_statistics(self, spikes_in_clusters):
        """Compute the statistics of all clusters."""

        nspikes, ndims = self.features.shape
        nclusters = len(spikes_in_clusters)
        LogP = np.zeros((nspikes, nclusters))
        stats = {}

        for c in spikes_in_clusters:
            # "my" refers to "my cluster"
            myspikes = spikes_in_clusters[c]
            myfeatures = np.take(self.y, myspikes, axis=0).astype(np.float64)
            nmyspikes = len(myfeatures)
            mymasks = np.take(self.masks, myspikes, axis=0)
            mymean = np.mean(myfeatures, axis=0).reshape((1, -1))
            # Boolean vector of size (nchannels,): which channels are unmasked?
            unmask = ((mymasks>0).sum(axis=0) > self.unmask_threshold)
            mask = ~unmask
            nunmask = np.sum(unmask)
            if nmyspikes <= 1 or nunmask == 0:
                mymean = np.zeros((1, myfeatures.shape[1]))
                covmat = 1e-3 * np.eye(nunmask)  # optim: nactivefeatures
                stats[c] = (mymean, covmat,
                            (1e-3)**ndims, nmyspikes,
                            np.zeros(ndims, dtype=np.bool)  # unmask
                            )
                continue

            # optimization: covmat only for submatrix of active features
            covmat = np.cov(myfeatures[:, unmask], rowvar=0) # stats for cluster c

            # Variation Bayesian approximation
            priorpoint = 1
            covmat *= (nmyspikes - 1)  # get rid of the normalization factor
            covmat += self.D[unmask, unmask] * priorpoint  # D = np.diag(sigma2.ravel())
            covmat /= (nmyspikes + priorpoint - 1)

            # the eta just for the current cluster
            etac = np.take(self.eta, myspikes, axis=0)
            # optimization: etac just for active features
            etac = etac[:, unmask]
            d = np.mean(etac, axis=0)

            # Handle nmasked == 0
            d[np.isnan(d)] = 0

            # add diagonal
            covmat += np.diag(d)

            # Compute the det of the covmat
            _sign, logdet = np.linalg.slogdet(covmat)
            if _sign < 0:
                warn("The correlation matrix of cluster %d has a negative determinant (whaaat??)" % c)

            stats[int(c)] = (mymean, covmat, logdet, nmyspikes, unmask)

        self.stats.update(stats)

    def compute_matrix(self, clusters, clusters_to_update=None):
        """Compute the correlation matrix between every pair of clusters.

        A dictionary pairs => value is returned.

        Compute all rows and columns corresponding to clusters_to_update.

        """
        nspikes, ndims = self.features.shape
        clusters_unique = np.unique(clusters)
        if clusters_to_update is None:
            clusters_to_update = clusters_unique

        # Indices of spikes in each cluster, for the clusters to update only.
        spikes_in_clusters = dict([(clu, np.nonzero(clusters == clu)[0])
                                   for clu in clusters_to_update])

        self.compute_cluster_statistics(spikes_in_clusters)
        stats = self.stats

        # New matrix (clu0, clu1) => new value
        C = {}

        def _compute_coeff(ci, cj):

            if ci not in stats or cj not in stats:
                C[ci, cj] = 0.
                return

            mui, Ci, logdeti, npointsi, unmaski = stats[ci]
            muj, Cj, logdetj, npointsj, unmaskj = stats[cj]

            if npointsi <= 1 or npointsj <= 1:
                C[ci, cj] = 0.
                return

            dmu = (muj - mui).reshape((-1, 1))

            unmasked = unmaskj
            masked = ~unmasked
            dmu_unmasked = dmu[unmasked]

            # pij is the probability that mui belongs to Cj:
            #    $$p_{ij} = w_j * N(\mu_i | \mu_j; C_j)$$
            # where wj is the relative size of cluster j
            # pii is the probability that mui belongs to Ci
            try:
                bj = np.linalg.solve(Cj, dmu_unmasked)
            except np.linalg.LinAlgError:
                bj = np.linalg.lstsq(Cj, dmu_unmasked)[0]

            var = np.sum(dmu[masked] ** 2 / self.sigma2[masked])
            logpij = (np.log(2*np.pi) * (-ndims/2.) +
                     -.5 * (logdetj + np.sum(np.log(self.sigma2[masked]))) +
                     -.5 * (np.dot(bj.T, dmu_unmasked) + var))

            # nspikes is the total number of spikes.
            wj = float(npointsj) / nspikes

            C[ci, cj] = wj * np.exp(logpij)[0,0]

        for ci in clusters_to_update:
            for cj in clusters_unique:
                _compute_coeff(ci, cj)
                _compute_coeff(cj, ci)

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

def normalize(matrix, direction='row'):

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
