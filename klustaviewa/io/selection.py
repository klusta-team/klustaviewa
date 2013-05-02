"""Functions for selecting portions of arrays."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Selection functions
# -----------------------------------------------------------------------------
def select_numpy(data, spikes):
    """Select a portion of an array with the corresponding spike indices.
    The first axis of data corresponds to the spikes."""
    
    if not hasattr(spikes, '__len__'):
        return data[spikes, ...]
    
    if type(spikes) == list:
        spikes = np.array(spikes)
    
    assert isinstance(data, np.ndarray)
    assert isinstance(spikes, np.ndarray)
    
    # spikes can contain boolean masks...
    if spikes.dtype == np.bool:
        data_selection = np.compress(spikes, data, axis=0)
    # or spike indices.
    else:
        data_selection = np.take(data, spikes, axis=0)
    return data_selection

def select_pandas(data, spikes, drop_empty_rows=True):
    
    if not hasattr(spikes, '__len__'):
        try:
            return np.array(data.ix[spikes]).squeeze()
        except KeyError:
            raise IndexError("Index {0:d} is not in the data.".format(
                spikes))
        
    try:
        # Remove empty rows.
        data_selected = data.ix[spikes]
    except IndexError:
        # This exception happens if the data is a view of the whole array,
        # and `spikes` is an array of booleans adapted to the whole array and 
        # not to the view. So we convert `spikes` into an array of indices,
        # so that Pandas can handle missing values.
        data_selected = data.ix[np.nonzero(spikes)[0]]
    if drop_empty_rows:
        data_selected = data_selected.dropna()
    return data_selected

def select(data, indices=None):
    """Select portion of the data, with the only assumption that indices are
    along the first axis.
    
    data can be a NumPy or Pandas object.
    
    """
    # indices=None or 'all' means select all.
    if indices is None or indices is 'all':
        return data
        
    indices_argument = indices
    if not hasattr(indices, '__len__'):
        indices = [indices]
        # is_single = True
    # else:
        # is_single = False
        
    # Ensure indices is an array of indices or boolean masks.
    if not isinstance(indices, np.ndarray):
        # Deal with empty indices.
        if not len(indices):
            if data.ndim == 1:
                return np.array([])
            elif data.ndim == 2:
                return np.array([[]])
            elif data.ndim == 3:
                return np.array([[[]]])
        else:
            if type(indices[0]) in (int, np.int32, np.int64):
                indices = np.array(indices, dtype=np.int32)
            elif type(indices[0]) == bool:
                indices = np.array(indices, dtype=np.bool)
            else:
                indices = np.array(indices)
    
    # Use NumPy or Pandas version
    if type(data) == np.ndarray:
        return select_numpy(data, indices_argument)
    else:
        return select_pandas(data, indices_argument)

def select_pairs(data, indices=None, conjunction='and'):
    """Select all items in data where at least one of the key index is in 
    indices.
    
    """
    if indices is None:
        return data
    assert isinstance(data, dict)
    if conjunction == 'and':
        return {(i, j): data[(i, j)] for (i, j) in data.keys() 
            if i in indices and j in indices}
    elif conjunction == 'or':
        return {(i, j): data[(i, j)] for (i, j) in data.keys() 
            if i in indices and j in indices}
        
def get_spikes_in_clusters(clusters_selected, clusters, return_indices=False):
    spike_indices = np.in1d(clusters, clusters_selected)
    if not return_indices:
        return spike_indices
    else:
        return np.nonzero(spike_indices)[0]
        
def get_some_spikes_in_clusters(clusters_selected, clusters,
        nspikes_max_expected=None,
        nspikes_per_cluster_min=None):
    """Select a sample of spikes among those belonging to the selected
    clusters, with at least `nspikes_per_cluster_min` spikes per cluster,
    and an expected maximum number of spikes equal to `nspikes_max_expected`.
    """
    if nspikes_max_expected is None:
        nspikes_max_expected = 100
    if nspikes_per_cluster_min is None:
        nspikes_per_cluster_min = 5
    
    spikes = []
    # Number of spikes in all selected clusters.
    counter = Counter(clusters)
    nspikes_in_clusters_selected = np.sum(np.array([counter[cluster]
        for cluster in clusters_selected]))
    # Take a sample of the spikes in each cluster.
    for cluster in clusters_selected:
        # Find the spike indices in the current cluster.
        spikes_in_cluster = get_spikes_in_clusters([cluster], clusters,
            return_indices=True)
        if len(spikes_in_cluster) == 0:
            continue
        nspikes_in_cluster = len(spikes_in_cluster)
        # Compute the number of spikes to select in this cluster.
        # This number is proportional to the relative size of the cluster,
        # so that large clusters have more spikes than small clusters.
        nspikes_in_cluster_requested = np.clip(int(
            nspikes_max_expected / float(nspikes_in_clusters_selected) * 
                len(spikes_in_cluster)),
            min(nspikes_per_cluster_min, nspikes_in_cluster),
            nspikes_in_cluster
            )
        # Choose randomly the appropriate number of spikes among those
        # belonging to the given cluster.
        spikes_selected = np.random.choice(spikes_in_cluster,
            nspikes_in_cluster_requested, replace=False)
        spikes.extend(spikes_selected)
    # Return the sorted array of all selected spikes.
    return np.array(sorted(spikes))

def get_some_spikes(clusters,
        nspikes_max=None,):
    """Select a sample of spikes, with a maximum number of spikes equal to 
    `nspikes_max`.
    """
    if nspikes_max is None:
        nspikes_max = 10000
    spikes = get_indices(clusters)
    spikes_selected = np.random.choice(spikes,
        min(nspikes_max, len(spikes)), replace=False)
    # Return the sorted array of all selected spikes.
    return np.array(sorted(spikes_selected))

def get_indices(data):
    if type(data) == np.ndarray:
        return np.arange(data.shape[0], dtype=np.int32)
    elif type(data) == pd.Series:
        return data.index
    elif type(data) == pd.DataFrame:
        return data.index
    elif type(data) == pd.Panel:
        return data.items
        
def to_array(data):
    """Convert a Pandas object to a NumPy array."""
    return np.atleast_1d(np.array(data).squeeze())
    
