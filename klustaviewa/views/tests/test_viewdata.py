"""Unit tests for the viewdata module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import time
import tempfile

import numpy as np

from klustaviewa.views.viewdata import *
from klustaviewa.views.tests.mock_data import (ncorrbins, corrbin,
        create_baselines, create_correlograms, create_similarity_matrix)
from klustaviewa.views.tests.utils import show_view
from klustaviewa.views import (WaveformView, FeatureView, ClusterView,
    CorrelogramsView, SimilarityMatrixView)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
# TODO: refactor this in proper mock data module in kwiklib
DIRPATH = tempfile.mkdtemp()
nchannels = 3

def rnd(*shape):
    return .25 * np.random.randn(*shape)
    
def rndint(*shape):
    return np.random.randint(size=shape, low=-32000, high=32000)

def setup():
    # Create files.
    prm = {'waveforms_nsamples': 10, 'nchannels': nchannels,
           'nfeatures_per_channel': 1,
           'sample_rate': 20000.,
           'duration': 10.}
    prb = {0:
        {
            'channels': range(nchannels),
            'graph': [(i, i+1) for i in range(nchannels-1)],
            'geometry': {i: [0., i] for i in range(nchannels)},
        }
    }
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=prm['sample_rate'],
                  start_time=10., 
                  start_sample=200000.,
                  bit_depth=16,
                  band_high=100.,
                  band_low=500.,
                  nchannels=nchannels,)
    add_event_type(files, 'myevents')
    add_cluster_group(files, name='Noise')
    add_cluster_group(files, name='MUA')
    add_cluster_group(files, name='Good')
    add_cluster_group(files, name='Unsorted')
    add_cluster(files, cluster_group=0, color=1)
    add_cluster(files, cluster_group=1, color=2)
    
    exp = Experiment(files=files)
    chgrp = exp.channel_groups[0]
    nspikes = 1000
    chgrp.spikes.time_samples.append(np.sort(rndint(nspikes)))
    chgrp.spikes.clusters.main.append(np.random.randint(size=nspikes, low=0, high=2))
    chgrp.spikes.features_masks.append(rnd(nspikes, nchannels, 2))
    chgrp.spikes.features_masks[..., 1] = chgrp.spikes.features_masks[..., 1] < .5
    chgrp.spikes.waveforms_raw.append(rndint(nspikes, 10, nchannels))
    chgrp.spikes.waveforms_filtered.append(rndint(nspikes, 10, nchannels))
    
    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_viewdata_waveformview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_waveformview_data(exp, clusters=[0, 1])
        show_view(WaveformView, **data)
    
def test_viewdata_featureview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_featureview_data(exp, clusters=[0], nspikes_bg=1000)
        show_view(FeatureView, **data)
    
def test_viewdata_clusterview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        data = get_clusterview_data(exp)
        show_view(ClusterView, **data)
    
def test_viewdata_correlogramsview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        correlograms = create_correlograms([0, 1], 50)
        data = get_correlogramsview_data(exp, clusters=[0, 1], 
                                         correlograms=correlograms,
                                         )
        show_view(CorrelogramsView, **data)
    
def test_viewdata_similaritymatrixview_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        chgrp = exp.channel_groups[0]
        matrix = create_similarity_matrix(2)
        data = get_similaritymatrixview_data(exp, matrix=matrix)
        show_view(SimilarityMatrixView, **data)
    
if __name__ == '__main__':
    setup()
    test_viewdata_featureview_1()
    teardown()
    