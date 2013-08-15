"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose.tools import with_setup

from kwiklib.dataio.tests.mock_data import (
                            nspikes, nclusters, nsamples, nchannels, fetdim)
from kwiklib.dataio import (HDF5Loader, HDF5Writer, select, get_indices,
    check_dtype, check_shape, get_array, load_text, KlustersLoader,
    klusters_to_hdf5
    )


def normalize_inplace(x):
    x -= np.mean(x)
    x /= np.abs(x.max())


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_hdf5_loader1():
    # Open the mock data.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    filename = os.path.join(dir, 'test.xml')
    
    global nspikes
    nspikes_total = nspikes
    
    # Convert in HDF5.
    klusters_to_hdf5(filename)
        
    # Open the file.
    filename_h5 = os.path.join(dir, 'test.klx')
    
    l = HDF5Loader(filename=filename_h5)
    lk = KlustersLoader(filename=filename)
    
    # Open probe.
    probe = l.get_probe()
    assert probe['nchannels'] == nchannels
    assert probe['nchannels_alive'] == nchannels
    assert np.array_equal(probe['channels'], np.arange(nchannels))
    
    # Select cluster.
    cluster = 3
    l.select(clusters=[cluster])
    lk.select(clusters=[cluster])
    
    # Get clusters.
    clusters = l.get_clusters('all')
    clusters_k = lk.get_clusters('all')
    nspikes = np.sum(clusters == cluster)
    
    # Check the clusters are correct.
    assert np.array_equal(get_array(clusters), get_array(clusters_k))
    
    # Get the spike times.
    spiketimes = l.get_spiketimes()
    spiketimes_k = lk.get_spiketimes()
    assert np.all(spiketimes <= 60)
    
    # Check the spiketimes are correct.
    assert np.allclose(get_array(spiketimes), get_array(spiketimes_k))
    
    # Get features.
    features = l.get_features()
    spikes = l.get_spikes()
    # Assert the indices in the features Pandas object correspond to the
    # spikes in the selected cluster.
    assert np.array_equal(features.index, spikes)
    # Assert the array has the right number of spikes.
    assert features.shape[0] == nspikes
    assert l.fetdim == fetdim
    assert l.nextrafet == 1
    
    # Get all features.
    features = l.get_features('all')
    features_k = lk.get_features('all')
    assert type(features) == pd.DataFrame
    assert features.shape[0] == nspikes_total
    
    # Check the features are correct.
    f = get_array(features)[:,:-1]
    f_k = get_array(features_k)[:,:-1]
    normalize_inplace(f)
    normalize_inplace(f_k)
    assert np.allclose(f, f_k, atol=1e-5)
    
    
    # Get masks.
    masks = l.get_masks('all')
    masks_k = lk.get_masks('all')
    assert masks.shape[0] == nspikes_total
    
    # Check the masks.
    assert np.allclose(masks.values, masks_k.values, atol=1e-2)
    
    # Get waveforms.
    waveforms = l.get_waveforms().values
    waveforms_k = lk.get_waveforms().values
    assert np.array_equal(waveforms.shape, (nspikes, nsamples, nchannels))
    
    # Check waveforms
    normalize_inplace(waveforms)
    normalize_inplace(waveforms_k)
    assert np.allclose(waveforms, waveforms_k, atol=1e-4)
    
    l.close()

def test_hdf5_save():
    # TODO
    # convert, make actions, save, load, check
    pass
    