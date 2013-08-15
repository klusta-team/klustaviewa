"""Unit tests for selection module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np
import pandas as pd

from kwiklib.dataio.selection import (select, select_pairs, get_spikes_in_clusters,
    to_array, get_some_spikes_in_clusters, get_some_spikes)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def generate_clusters(indices, nspikes=100):
    """Generate all spikes in cluster 0, except some in cluster 1."""
    # 2 different clusters, with 3 spikes in cluster 1
    clusters = np.zeros(nspikes, dtype=np.int32)
    clusters[indices] = 1
    return clusters
    
def generate_data2D(nspikes=100, ncols=5):
    data = np.random.randn(nspikes, ncols)
    return data
    
def test_cluster_selection():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    clusters_selected = [1, 2]
    spikes = get_spikes_in_clusters(clusters_selected, clusters, False)
    assert np.array_equal(np.nonzero(spikes)[0], indices)
    spikes = get_spikes_in_clusters(clusters_selected, clusters, True)
    assert np.array_equal(spikes, indices)

def test_select_numpy():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])

def test_select_pairs():
    indices = [10, 20, 25]
    clusters = {(i, j): (i + j) for i in xrange(30) for j in xrange(30) if i <= j}
    pairs_selected = select_pairs(clusters, indices)
    assert len(pairs_selected) == 6

def test_select_pandas():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    
    # test selection of Series (1D)
    clusters = pd.Series(clusters)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])
    
    # test selection of Series (3D)
    clusters = pd.DataFrame(clusters)
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test selection of Panel (4D)
    clusters = pd.Panel(np.expand_dims(clusters, 3))
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test recursive selection
    assert np.array_equal(to_array(select(select(clusters, [10, 25]), 25)), [1])

def test_select_single():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert select(clusters, 10) == 1

def test_select_some():
    nspikes = 100
    clusters = np.zeros(nspikes, dtype=np.int32)
    for i in xrange(1, 10):
        clusters[range(i, nspikes, 10)] = i
    clusters_selected = [2, 3, 5]
    nspikes_max_expected = 20
    nspikes_per_cluster_min = 2
    spikes = get_some_spikes_in_clusters(clusters_selected, clusters,
        counter=Counter(clusters),
        nspikes_max_expected=nspikes_max_expected,
        nspikes_per_cluster_min=nspikes_per_cluster_min)
    
    assert len(spikes) >= nspikes_per_cluster_min * len(clusters_selected)
    
    spikes = get_some_spikes(clusters, 10)
    
    assert len(np.arange(spikes.start, spikes.stop, spikes.step)) == 10
    
def test_select_array():
    # All spikes in cluster 1.
    indices = [10, 20, 25]
    
    # Indices in data excerpt.
    indices_data = [5, 10, 15, 20]
    
    # Generate clusters and data.
    clusters = generate_clusters(indices)
    data_raw = generate_data2D()
    data = pd.DataFrame(data_raw)
    
    # Excerpt of the data.
    data_excerpt = select(data, indices_data)
    
    # Get all spike indices in cluster 1.
    spikes_inclu1 = get_spikes_in_clusters([1], clusters)
    
    # We want to select all clusters in cluster 1 among those in data excerpt.
    data_excerpt_inclu1 = select(data_excerpt, spikes_inclu1)
    
    # There should be two rows: 4 in the excerpt, among which two are in
    # cluster 1.
    assert data_excerpt_inclu1.shape == (2, 5)
    