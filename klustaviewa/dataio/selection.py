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
    
    if isinstance(spikes, slice):
        return np.array(data.iloc[spikes]).squeeze()
    elif not hasattr(spikes, '__len__'):
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

def pandaize(values, spikes):
    """Convert a NumPy array to a Pandas object, with the spikes indices."""
    # Get the spike indices.
    if isinstance(spikes, slice):
        spike_indices = np.arange(spikes.start, spikes.stop, spikes.step)
    elif spikes.dtype == np.bool:
        spike_indices = np.nonzero(spikes)[0]
    else:
        spike_indices = spikes
    
    # Create the Pandas object with the spike indices.
    if values.ndim == 1:
        pd_arr = pd.Series(values, index=spike_indices)
    elif values.ndim == 2:
        pd_arr = pd.DataFrame(values, index=spike_indices)
    elif values.ndim == 3:
        pd_arr = pd.Panel(values, items=spike_indices)
    return pd_arr
    
def select_pytables(data, spikes):
    if len(data) == 2:
        table, column = data
        process_fun = None
    elif len(data) == 3:
        table, column, process_fun = data
    values = table[spikes][column]
    # Process the NumPy array.
    if process_fun:
        values = process_fun(values)
    return pandaize(values, spikes)
    
def select(data, indices=None):
    """Select portion of the data, with the only assumption that indices are
    along the first axis.
    
    data can be a NumPy or Pandas object.
    
    """
    # indices=None or 'all' means select all.
    if indices is None or indices is 'all':
        if type(data) == tuple:
            indices = np.ones(data[0].shape[0], dtype=np.bool)
        else:
            return data
        
    indices_argument = indices
    if not hasattr(indices, '__len__') and not isinstance(indices, slice):
        indices = [indices]
        
    # Ensure indices is an array of indices or boolean masks.
    if not isinstance(indices, np.ndarray) and not isinstance(indices, slice):
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
    
    # Use NumPy, PyTables (tuple) or Pandas version
    if type(data) == np.ndarray:
        if data.size == 0:
            return data
        return select_numpy(data, indices_argument)
    elif type(data) == tuple:
        return select_pytables(data, indices_argument)
    else:
        if data.values.size == 0:
            return data
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
    
def get_some_spikes_in_clusters(clusters_selected, clusters, counter=None,
        nspikes_max_expected=None,
        nspikes_per_cluster_min=None):
    """Select a sample of spikes among those belonging to the selected
    clusters, with at least `nspikes_per_cluster_min` spikes per cluster,
    and an expected maximum number of spikes equal to `nspikes_max_expected`.
    """
    if not hasattr(clusters_selected, '__len__'):
        clusters_selected = [clusters_selected]
    if nspikes_max_expected is None:
        nspikes_max_expected = 100
    if nspikes_per_cluster_min is None:
        nspikes_per_cluster_min = 5
    
    nspikes = len(clusters)
    spikes = np.zeros(nspikes, dtype=np.bool)
    # Number of spikes in all selected clusters.
    nspikes_in_clusters_selected = np.sum(np.array([counter[cluster]
        for cluster in clusters_selected]))
    # Take a sample of the spikes in each cluster.
    nclusters_selected = len(clusters_selected)
    s = np.zeros(nspikes, dtype=np.bool)
    for cluster in clusters_selected:
        # Find the spike indices in the current cluster.
        spikes_in_cluster = get_spikes_in_clusters([cluster], clusters)
        spikes_in_cluster0 = spikes_in_cluster.copy()
        # Discard empty clusters.
        if not(np.any(spikes_in_cluster)):
            continue
        nspikes_in_cluster = np.sum(spikes_in_cluster)
        # Compute the number of spikes to select in this cluster.
        # This number is proportional to the relative size of the cluster,
        # so that large clusters have more spikes than small clusters.
        try:
            nspikes_in_cluster_requested = np.clip(int(
                nspikes_max_expected / float(nclusters_selected)),
                min(nspikes_per_cluster_min, nspikes_in_cluster),
                nspikes_in_cluster
                )
        except OverflowError:
            nspikes_in_cluster_requested = nspikes_in_cluster
        # Choose randomly the appropriate number of spikes among those
        # belonging to the given cluster.
        # Probability to take each spike.
        p = nspikes_in_cluster_requested / float(nspikes_in_cluster)
        if p > 0:
            # Remove evenly distributed spikes so that the expected number
            # of selected spikes in that cluster is approximately 
            # nspikes_in_cluster_requested.
            k = max(int(1. / p), 1)
            for _ in xrange(10):
                s[:] = False
                # s[np.random.randint(low=0, high=k)::k] = True
                s[0::k] = True
                spikes_in_cluster = spikes_in_cluster0 & s
                # Try to increase the number of spikes if there are no
                # spikes in the current cluster.
                if not any(spikes_in_cluster):
                    k = max(1, k // 2)
                else:
                    break
        spikes = spikes | spikes_in_cluster
    
    # Return the sorted array of all selected spikes.
    return np.nonzero(spikes)[0]

def get_some_spikes(clusters,
        nspikes_max=None,):
    """Select a sample of spikes, with a maximum number of spikes equal to 
    `nspikes_max`.
    """
    if nspikes_max is None:
        nspikes_max = 10000
    spikes = get_indices(clusters)
    p = nspikes_max / float(len(spikes))
    k = max(int(1. / p), 1)
    return slice(0, len(spikes), k)
    
def get_indices(data):
    # Convert list to array.
    if type(data) == list:
        data = np.array(data)
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
    
