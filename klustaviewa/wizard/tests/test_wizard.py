"""Unit tests for wizard module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import OrderedDict

from nose.tools import raises
import pandas as pd
import numpy as np

from klustaviewa.wizard.wizard import Wizard
from kwiklib.utils import logger as log
from kwiklib.dataio.tests.mock_data import (
    nspikes, nclusters, nsamples, nchannels, fetdim, cluster_offset,
    create_clusters, create_similarity_matrix)

    
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def create_cluster_groups(nclusters):
    return pd.Series(np.array(np.ones(nclusters) * 3, dtype=np.int32),
        index=np.arange(nclusters) + cluster_offset)
    

# -----------------------------------------------------------------------------
# Wizard tests
# -----------------------------------------------------------------------------
def test_wizard():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    best_cluster = clusters_unique[np.argmax(quality)]
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    w.update_candidates()
    
    # Check the first target cluster.
    assert w.current_target() == best_cluster
    
    # Test impossible previous.
    assert w.previous_candidate() == w.current_candidate()
    
    # Check next/previous.
    c0 = w.next_candidate()
    c1 = w.next_candidate()
    assert w.previous_candidate() == c0
    assert w.next_candidate() == c1
    
    # Check skip target.
    t0 = w.current_target()
    w.skip_target()
    w.update_candidates()
    t1 = w.current_target()
    assert t0 != t1
    
def test_wizard_merge():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    target = clusters_unique[np.argmax(quality)]
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    w.update_candidates()
    
    cluster = w.current_candidate()
    
    # Simulate a merge: target and cluster ==> cluster_new.
    cluster_new = clusters_unique.max() + 1
    clusters[clusters == target] = cluster_new
    clusters[clusters == cluster] = cluster_new
    log.debug("Merged {0:d} and {1:d} to {2:d}".format(
        target, cluster, cluster_new))
    similarity_matrix = create_similarity_matrix(nclusters - 1)
    indices = [x for x in xrange(cluster_offset, cluster_offset + nclusters + 1)
                    if x != cluster and x != target]
    cluster_groups = pd.Series(np.array(np.ones(nclusters - 1) * 3, dtype=np.int32),
        index=np.array(indices))
    
    # Update the wizard.
    quality = np.diag(similarity_matrix)
    w.set_data(similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    w.update_candidates(cluster_new)
    
    assert w.current_target() == cluster_new
    
    c = w.current_candidate()
    assert c is not None
    assert w.previous_candidate() == w.current_candidate()
    assert w.next_candidate() == c
    
    for _ in xrange(nclusters):
        c = w.next_candidate()
        assert c not in (target, cluster)
    
def test_wizard_move():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    target = clusters_unique[np.argmax(quality)]
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    w.update_candidates()
    
    cluster0 = w.current_candidate()
    cluster1 = w.next_candidate()
    cluster2 = w.next_candidate()
    
    # Simulate a move.
    cluster_groups.ix[cluster2] = 1
    
    # Update the wizard.
    w.set_data(cluster_groups=cluster_groups)
    w.update_candidates(target)
    
    assert w.current_target() == target
    assert w.current_candidate() not in (cluster0, cluster1, cluster2)
    
    for _ in xrange(nclusters):
        c = w.next_candidate()
        assert c != cluster2
    
    